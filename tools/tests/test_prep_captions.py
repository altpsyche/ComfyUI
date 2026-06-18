"""Unit tests for the caption-pruning logic (pure functions, no filesystem)."""
import subprocess
import sys
from pathlib import Path

import prep_captions as P

PREP = Path(P.__file__).resolve()


def _run(tmp, *args):
    return subprocess.run([sys.executable, str(PREP), str(tmp), *args],
                          capture_output=True, text=True)


def test_norm_plural_fold():
    assert P._norm("shoes") == "shoe"
    assert P._norm("wristbands") == "wristband"
    assert P._norm("gown") == "gown"
    assert P._norm("dress") == "dres"          # documented quirk: self-consistent both sides


def test_build_lock_expands_synonym_cluster():
    _, nouns = P.build_lock("white shoes", "")
    assert {"shoe", "boot", "sneaker", "footwear"} <= nouns   # footwear cluster pulled in


def test_should_prune_outfit_headnoun_and_variants():
    phrases, nouns = P.build_lock("teal and white tennis dress", "")
    assert P.should_prune("white dress", phrases, nouns, set())
    assert P.should_prune("tennis dress", phrases, nouns, set())
    assert P.should_prune("dress", phrases, nouns, set())


def test_protect_set_and_keep_override():
    phrases, nouns = P.build_lock("white shoes", "")
    assert not P.should_prune("looking at viewer", phrases, nouns, set())   # PROTECT
    assert not P.should_prune("white shoes", phrases, nouns, {"white shoes"})  # explicit keep wins


def test_keep_makes_garment_promptable():
    # the overcoat case: a kept garment stays promptable (add/removable at inference), not baked.
    phrases, nouns = P.build_lock("long coat, turtleneck sweater", "")
    assert P.should_prune("coat", phrases, nouns, set())          # baked by default (head-noun match)
    assert not P.should_prune("coat", phrases, nouns, {"coat"})   # in keep -> stays a promptable tag


def test_structural_tags_never_locked():
    phrases, _ = P.build_lock("1girl, solo, tennis dress", "")
    assert "1girl" not in phrases and "solo" not in phrases


def test_underscored_wd14_tags_are_matched():
    # WD14 emits underscored tags (crop_top); the outfit string uses spaces. Pruning must still fire,
    # else the outfit never bakes into the trigger and stays promptable.
    phrases, nouns = P.build_lock("crop top, ripped denim shorts, open shirt", "")
    assert P.should_prune("crop_top", phrases, nouns, set())
    assert P.should_prune("denim_shorts", phrases, nouns, set())
    assert P.should_prune("open_shirt", phrases, nouns, set())
    assert P.should_prune("white_shirt", phrases, nouns, set())      # shirt -> top cluster
    assert not P.should_prune("looking_at_viewer", phrases, nouns, set())  # PROTECT, underscored


def test_trigger_not_pruned_by_headnoun():
    # a non-outfit tag with an unrelated head noun stays promptable
    phrases, nouns = P.build_lock("tennis uniform", "")
    assert not P.should_prune("green eyes", phrases, nouns, set())


# --- the zero-bake guard (the silent-failure mode) ---

def test_strict_fails_on_zero_bake(tmp_path):
    (tmp_path / "a.txt").write_text("tennis_uniform, visor, 1girl", encoding="utf-8")
    r = _run(tmp_path, "--trigger", "zz", "--outfit", "ballgown, tiara", "--strict")
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "did NOT" in r.stderr


def test_warn_only_without_strict(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("tennis_uniform, visor, 1girl", encoding="utf-8")
    r = _run(tmp_path, "--trigger", "zz", "--outfit", "ballgown")
    assert r.returncode == 0                      # warns but does not fail
    assert f.read_text(encoding="utf-8").startswith("zz")   # still prepped


def test_strict_passes_when_outfit_matches(tmp_path):
    (tmp_path / "a.txt").write_text("tennis_uniform, visor, 1girl", encoding="utf-8")
    r = _run(tmp_path, "--trigger", "zz", "--outfit", "tennis uniform, visor", "--strict")
    assert r.returncode == 0, r.stderr            # underscored tags matched -> baked -> ok


# --- modular character: a two-token trigger (identity + per-outfit token), keep_tokens=2 ---

def test_two_token_trigger_prepends_both_and_bakes(tmp_path):
    # A modular outfit subfolder: identity token + this outfit's token lead every caption; the identity
    # (--prune) and this outfit's garments (--outfit) are removed, the rest stays promptable.
    f = tmp_path / "a.txt"
    f.write_text("black hair, hoodie, pleated skirt, looking at viewer, outdoors", encoding="utf-8")
    r = _run(tmp_path, "--trigger", "mirachar, mira_hoodie",
             "--prune", "black hair", "--outfit", "hoodie, pleated skirt", "--strict")
    assert r.returncode == 0, r.stderr
    out = f.read_text(encoding="utf-8")
    assert out.startswith("mirachar, mira_hoodie, ")            # both trigger tokens kept first (keep_tokens=2)
    tags = [t.strip() for t in out.split(",")]
    assert "black hair" not in tags                             # identity baked
    assert "hoodie" not in tags and "pleated skirt" not in tags  # this outfit's garments baked
    assert "looking at viewer" in tags and "outdoors" in tags    # pose/scene stays promptable


def test_two_token_trigger_no_double_prepend_on_rerun(tmp_path):
    # Re-running prep on already-prepped captions must not double-prepend either trigger token.
    f = tmp_path / "a.txt"
    f.write_text("mirachar, mira_hoodie, outdoors", encoding="utf-8")
    r = _run(tmp_path, "--trigger", "mirachar, mira_hoodie")
    assert r.returncode == 0, r.stderr
    assert f.read_text(encoding="utf-8") == "mirachar, mira_hoodie, outdoors"

"""The build emits the expected graph set, roster matches the roster, and every graph validates."""
import json
import sys

import pytest

import il_graphs.build as B
from il_graphs.config import ROOT, CHARACTERS
from validate_workflow import validate

STATIC = ["IL_1_Base", "IL_2_Refine", "IL_3_Guided", "IL_4_Studio", "IL_5_Max",
          "IL_IPAdapter", "IL_Pose", "IL_LCM", "IL_XYPlot"]


def _expected_names():
    # A modular character (a `outfits` table) emits one graph PER OUTFIT; everyone else emits one.
    names = list(STATIC)
    for c, spec in CHARACTERS.items():
        if "outfits" in spec:
            names += [f"IL_DatasetEdit_{c}_{o}" for o in spec["outfits"]]
        else:
            names.append(f"IL_DatasetEdit_{c}")
    return names


def _build_with(characters, monkeypatch, tmp_path):
    """Run build.main() against a substitute roster, writing to an ISOLATED tmp dir (so a passing build
    never pollutes the real workflows/roster). Raises from validation propagate to the caller."""
    monkeypatch.setattr(B, "CHARACTERS", characters)
    monkeypatch.setattr(B, "OUT", tmp_path)
    monkeypatch.setattr(B, "ROOT", tmp_path)
    (tmp_path / "tools" / "lora_train").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["build", "--no-validate"])
    B.main()
    return tmp_path


def test_roster_matches_characters(built):
    roster = json.loads((ROOT / "tools/lora_train/roster.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in roster] == list(CHARACTERS)
    for e in roster:                                  # roster schema the trainer relies on
        assert set(e) >= {"name", "trigger", "id", "outfit", "prune"}


def test_expected_graphs_emitted(built):
    for name in _expected_names():
        assert (built / f"{name}.json").exists(), f"missing {name}.json"


def test_all_graphs_validate(built):
    for name in _expected_names():
        rc = validate(built / f"{name}.json", require_models=False, require_wildcards=True)
        assert rc == 0, f"{name} failed validation (rules/wildcards)"


def test_like_variants_inherit_id_seed_prune(built):
    """A `like` variant inherits the parent's id + prune unless it overrides them."""
    roster = {e["name"]: e for e in
              json.loads((ROOT / "tools/lora_train/roster.json").read_text(encoding="utf-8"))}
    likes = [(n, s) for n, s in CHARACTERS.items() if "like" in s]
    if not likes:
        pytest.skip("no `like` variant in the roster (all same-identity pairs are modular)")
    for name, spec in likes:
        parent = CHARACTERS[spec["like"]]
        if "id" not in spec:
            assert roster[name]["id"] == parent["id"], f"{name} did not inherit id"
        if "prune" not in spec:
            assert roster[name]["prune"] == parent.get("prune", ""), f"{name} did not inherit prune"


# --- modular characters (a `[<char>.outfits]` table -> one LoRA, swappable per-outfit tokens) ---

def test_modular_roster_and_per_outfit_graphs(built):
    roster = {e["name"]: e for e in
              json.loads((ROOT / "tools/lora_train/roster.json").read_text(encoding="utf-8"))}
    modular = {n: s for n, s in CHARACTERS.items() if "outfits" in s}
    assert modular, "no modular character in the roster to exercise the feature"
    for name, spec in modular.items():
        e = roster[name]
        assert e["outfits"] == dict(spec["outfits"])             # roster carries the outfits map
        assert e["outfit"] == "" and e["keep"] == ""             # single-outfit fields empty for modular
        for outfit in spec["outfits"]:
            assert (built / f"IL_DatasetEdit_{name}_{outfit}.json").exists()
        assert not (built / f"IL_DatasetEdit_{name}.json").exists()   # no bare graph for a modular char


def test_modular_rejects_outfit_or_like(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _build_with({"x": {"id": "1girl, solo", "outfit": "dress", "outfits": {"a": "hat"}}}, monkeypatch, tmp_path)


def test_modular_outfit_name_must_be_lowercase_alnum(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _build_with({"x": {"id": "1girl, solo", "outfits": {"bad_name": "hat"}}}, monkeypatch, tmp_path)


def test_modular_keep_must_be_subset_of_outfits(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _build_with({"x": {"id": "1girl, solo", "outfits": {"a": "hat"}, "keep": {"b": "hat"}}}, monkeypatch, tmp_path)


def test_keep_table_without_outfits_raises(monkeypatch, tmp_path):
    # renaming [x.outfits] -> [x.keep] (or dropping `outfits`) must fail loudly, not silently drop outfits
    with pytest.raises(ValueError):
        _build_with({"x": {"id": "1girl, solo", "keep": {"a": "hat"}}}, monkeypatch, tmp_path)


def test_graph_name_collision_raises(monkeypatch, tmp_path):
    # modular 'mira' outfit 'winter' and a separate top-level 'mira_winter' both -> IL_DatasetEdit_mira_winter
    bad = {"mira": {"id": "1girl, solo", "outfits": {"winter": "coat"}},
           "mira_winter": {"id": "1girl, solo"}}
    with pytest.raises(ValueError):
        _build_with(bad, monkeypatch, tmp_path)


def test_like_inheritance_resolves(monkeypatch, tmp_path):
    """The `like` mechanism still works (kept covered even though no roster entry uses it right now)."""
    chars = {"base": {"id": "1girl, solo, redhead", "outfit": "dress", "prune": "redhead", "hero_seed": 7},
             "var": {"like": "base", "outfit": "gown"}}                       # inherits id+prune, overrides outfit
    out = _build_with(chars, monkeypatch, tmp_path)
    roster = {e["name"]: e for e in
              json.loads((out / "tools/lora_train/roster.json").read_text(encoding="utf-8"))}
    assert roster["var"]["id"] == "1girl, solo, redhead"     # inherited
    assert roster["var"]["prune"] == "redhead"               # inherited
    assert roster["var"]["outfit"] == "gown"                 # own override
    assert (out / "IL_DatasetEdit_var.json").exists()


def test_like_cannot_point_at_modular_parent(monkeypatch, tmp_path):
    bad = {"m": {"id": "1girl, solo", "outfits": {"a": "hat"}},
           "v": {"like": "m", "outfit": "dress"}}
    with pytest.raises(ValueError):
        _build_with(bad, monkeypatch, tmp_path)

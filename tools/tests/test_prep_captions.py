"""Unit tests for the caption-pruning logic (pure functions, no filesystem)."""
import prep_captions as P


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


def test_structural_tags_never_locked():
    phrases, _ = P.build_lock("1girl, solo, tennis dress", "")
    assert "1girl" not in phrases and "solo" not in phrases


def test_trigger_not_pruned_by_headnoun():
    # a non-outfit tag with an unrelated head noun stays promptable
    phrases, nouns = P.build_lock("tennis uniform", "")
    assert not P.should_prune("green eyes", phrases, nouns, set())

"""Unit tests for gen_dataset's pure seed-randomizer (no network, no filesystem)."""
import random

import gen_dataset as G


def _prompt():
    return {
        "1": {"class_type": "KSampler", "inputs": {"seed": 5, "cfg": 1.0, "steps": 6}},
        "2": {"class_type": "ImpactWildcardProcessor", "inputs": {"seed": 42, "wildcard_text": "x"}},
        "3": {"class_type": "FaceDetailer", "inputs": {"seed": ["9", 0], "denoise": 0.35}},  # linked -> skip
        "4": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 7}},
    }


def test_randomize_changes_only_unlinked_seeds():
    p = _prompt()
    changed = G.randomize_seeds(p, random.Random(123))
    assert changed == 3                               # two `seed` widgets + one `noise_seed`
    assert p["3"]["inputs"]["seed"] == ["9", 0]       # linked seed untouched
    assert p["1"]["inputs"]["cfg"] == 1.0             # non-seed inputs untouched
    assert p["2"]["inputs"]["wildcard_text"] == "x"
    for nid, key in (("1", "seed"), ("2", "seed"), ("4", "noise_seed")):
        v = p[nid]["inputs"][key]
        assert isinstance(v, int) and 0 <= v <= G.SEED_MAX


def test_randomize_is_deterministic_for_a_given_base_seed():
    a, b = _prompt(), _prompt()
    G.randomize_seeds(a, random.Random(999))
    G.randomize_seeds(b, random.Random(999))
    assert [a[n]["inputs"].get("seed", a[n]["inputs"].get("noise_seed")) for n in ("1", "2", "4")] == \
           [b[n]["inputs"].get("seed", b[n]["inputs"].get("noise_seed")) for n in ("1", "2", "4")]


def test_no_seed_inputs_returns_zero():
    assert G.randomize_seeds({"1": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0]}}},
                             random.Random(1)) == 0

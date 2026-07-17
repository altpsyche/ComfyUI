"""Unit tests for the modular multi-subset dataset planner (pure functions, no filesystem)."""
import dataset_plan as D


def test_balanced_repeats_equalizes_uneven_outfits():
    # Two outfits, 20 vs 40 frames, target 2000 steps / 4 epochs / batch 2 -> each gets an equal share
    # (1000 steps): 1000*2/(20*4)=25 and 1000*2/(40*4)=12.5->12. The smaller outfit repeats MORE so it
    # contributes the same training signal -- no outfit dominates (red flag #6).
    r = D.balanced_repeats([20, 40], steps=2000, epochs=4, batch=2)
    assert r == [25, 12]
    # the per-outfit step-share is ~equal despite the 2x frame gap
    share = [c * rep / 2 * 4 for c, rep in zip([20, 40], r)]   # images*repeats*epochs/batch (rearranged)
    assert abs(share[0] - share[1]) / max(share) < 0.10


def test_balanced_repeats_floors_at_one():
    # A huge outfit relative to the budget still trains at least once per frame.
    assert D.balanced_repeats([10000], steps=100, epochs=4, batch=2) == [1]


def test_balanced_repeats_rejects_empty_subset():
    try:
        D.balanced_repeats([20, 0], steps=2000, epochs=4, batch=2)
        assert False, "expected ValueError on a zero-image subset"
    except ValueError:
        pass


CFG = {"resolution": 1024, "batch": 2, "min_bucket_reso": 768,
       "max_bucket_reso": 1280, "bucket_reso_steps": 64}


def test_render_dataset_toml_keep_tokens_and_subsets():
    subsets = [{"image_dir": r"C:\d\mira\hoodie", "num_repeats": 25},
               {"image_dir": "C:/d/mira/winter", "num_repeats": 12}]
    toml = D.render_dataset_toml("mira", subsets, CFG, keep_tokens=2)
    assert "keep_tokens = 2" in toml
    assert toml.count("[[datasets.subsets]]") == 2
    assert 'image_dir = "C:/d/mira/hoodie"' in toml      # backslashes normalized to forward slashes
    assert 'image_dir = "C:/d/mira/winter"' in toml
    assert "num_repeats = 25" in toml and "num_repeats = 12" in toml
    assert "resolution = 1024" in toml and "batch_size = 2" in toml


def test_render_dataset_toml_parses_as_valid_toml():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    subsets = [{"image_dir": "C:/d/mira/hoodie", "num_repeats": 25},
               {"image_dir": "C:/d/mira/winter", "num_repeats": 12}]
    doc = tomllib.loads(D.render_dataset_toml("mira", subsets, CFG, keep_tokens=2))
    assert doc["general"]["keep_tokens"] == 2
    assert len(doc["datasets"][0]["subsets"]) == 2


def test_render_dataset_toml_reg_subset():
    subsets = [{"image_dir": "C:/d/mira/hoodie", "num_repeats": 25},
               {"image_dir": "C:/d/reg/1girl", "num_repeats": 1, "is_reg": True, "class_tokens": "1girl"}]
    toml = D.render_dataset_toml("mira", subsets, CFG, keep_tokens=2)
    assert "is_reg = true" in toml and 'class_tokens = "1girl"' in toml


def test_render_dataset_toml_rejects_empty():
    try:
        D.render_dataset_toml("mira", [], CFG, keep_tokens=2)
        assert False, "expected ValueError on no subsets"
    except ValueError:
        pass

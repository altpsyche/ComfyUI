"""Regression for R1: the CFG floor is enforced on the real sampler nodes, not just CFGGuider."""
from validate_workflow import check_rules

CFG_IDX = {"KSampler": 3, "KSamplerAdvanced": 4, "UltimateSDUpscale": 4,
           "FaceDetailer": 6, "SEGSDetailer": 6, "CFGGuider": 0}


def _graph(cfg, ntype="KSampler"):
    idx = CFG_IDX[ntype]
    wv = [0] * (idx + 1)
    wv[idx] = cfg
    return {"nodes": [{"id": 1, "type": ntype, "mode": 0, "widgets_values": wv}]}


def test_ksampler_below_min_fails():
    errs = check_rules(_graph(3.0), {"min_cfg": 5.0})
    assert errs and "cfg = 3.0" in errs[0]


def test_ksampler_at_min_passes():
    assert check_rules(_graph(5.0), {"min_cfg": 5.0}) == []


def test_low_cfg_passes_when_no_min_rule():
    # LCM / Qwen-Edit graphs: no min_cfg rule -> low cfg is fine.
    assert check_rules(_graph(1.5), {"clip_skip": -2}) == []


def test_usdu_and_detailers_are_checked():
    for nt in ("UltimateSDUpscale", "FaceDetailer", "SEGSDetailer"):
        assert check_rules(_graph(3.0, nt), {"min_cfg": 5.0}), f"{nt} cfg not checked"


def test_max_cfg_enforced():
    assert check_rules(_graph(15.0), {"max_cfg": 12.0})


def test_bypassed_node_skipped():
    g = _graph(3.0)
    g["nodes"][0]["mode"] = 4            # muted/bypassed
    assert check_rules(g, {"min_cfg": 5.0}) == []

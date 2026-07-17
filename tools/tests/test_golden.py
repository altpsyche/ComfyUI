"""Lock the generated graph JSON: a refactor must not change any emitted .json.

Fixtures in tests/golden/ were captured from the pre-refactor build. If a future change is meant to
alter a graph, regenerate the golden on purpose (re-copy from user/default/workflows/).
"""
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden"

# IL_XYPlot is intentionally NOT golden-locked: it bakes an absolute, machine-specific LoRA-batch path
# (required by the efficiency-nodes LoRA-Batch loader), so byte-equality isn't portable. test_build
# still asserts it is emitted and validates.
EXCLUDE = {"IL_XYPlot"}


def test_graphs_byte_identical_to_golden(built):
    goldens = sorted(gf for gf in GOLDEN.glob("*.json") if gf.stem not in EXCLUDE)
    assert goldens, "no golden fixtures captured"
    for gf in goldens:
        live = built / gf.name
        assert live.exists(), f"golden {gf.name} has no live counterpart"
        assert live.read_text(encoding="utf-8") == gf.read_text(encoding="utf-8"), \
            f"{gf.name} differs from golden — graph output changed unexpectedly"

"""The build emits the expected graph set, roster matches the roster, and every graph validates."""
import json

from il_graphs.config import ROOT, CHARACTERS
from validate_workflow import validate

STATIC = ["IL_1_Base", "IL_2_Refine", "IL_3_Guided", "IL_4_Studio", "IL_5_Max",
          "IL_IPAdapter", "IL_Pose", "IL_LCM", "IL_XYPlot"]


def _expected_names():
    return STATIC + [f"IL_DatasetEdit_{c}" for c in CHARACTERS]


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

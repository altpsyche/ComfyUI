"""Keep docs honest: every `dev train` flag + every train.toml profile must be in REFERENCE.md."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LORA = ROOT / "tools" / "lora_train"
REF = (LORA / "REFERENCE.md").read_text(encoding="utf-8")
TOML = (LORA / "train.toml").read_text(encoding="utf-8")
TRAIN = (ROOT / "devtools" / "train" / "__init__.py").read_text(encoding="utf-8")

# Generic dispatcher flags documented elsewhere (not part of the training reference table).
_IGNORE = {"--help"}


def _train_flags():
    return sorted(set(re.findall(r'add_argument\(\s*"(--[a-z][\w-]*)"', TRAIN)) - _IGNORE)


def test_every_flag_documented():
    missing = [f for f in _train_flags() if f not in REF]
    assert not missing, f"flags missing from REFERENCE.md: {missing}"


def test_every_profile_documented():
    profiles = re.findall(r"\[profiles\.(\w+)\]", TOML)
    assert profiles, "no profiles found in train.toml"
    missing = [p for p in profiles if p not in REF]
    assert not missing, f"profiles missing from REFERENCE.md: {missing}"

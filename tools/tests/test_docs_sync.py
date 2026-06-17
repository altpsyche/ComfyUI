"""Keep docs honest: every train_lora flag + every train.toml profile must be in REFERENCE.md."""
import re
from pathlib import Path

LORA = Path(__file__).resolve().parents[1] / "lora_train"
REF = (LORA / "REFERENCE.md").read_text(encoding="utf-8")
PS = (LORA / "train_lora.ps1").read_text(encoding="utf-8")
TOML = (LORA / "train.toml").read_text(encoding="utf-8")


def _param_flags():
    block = re.search(r"\nparam\((.*?)\n\)", PS, re.S).group(1)
    return sorted(set(re.findall(r"\$([A-Za-z]\w*)", block)))


def test_every_flag_documented():
    missing = [f for f in _param_flags() if f"-{f}" not in REF]
    assert not missing, f"flags missing from REFERENCE.md: {missing}"


def test_every_profile_documented():
    profiles = re.findall(r"\[profiles\.(\w+)\]", TOML)
    assert profiles, "no profiles found in train.toml"
    missing = [p for p in profiles if p not in REF]
    assert not missing, f"profiles missing from REFERENCE.md: {missing}"

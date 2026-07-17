"""Regression for W5: the param precedence chain and that defaults reproduce today's values."""
import pytest

import train_config as T

DOC = {
    "defaults": {"dim": 16, "alpha": 8, "steps": 1500, "optimizer": "prodigy", "d_coef": 1.0, "lr": ""},
    "profiles": {"fast": {"dim": 8, "steps": 800}, "complex": {"dim": 32, "steps": 2000, "d_coef": 0.9}},
    "train": {"nyx": {"dim": 24, "steps": 1234}},
}


def r(**kw):
    return T.resolve(doc=DOC, **kw)


def test_defaults_only():
    assert r(char="aria")["dim"] == 16


def test_per_char_beats_defaults():
    c = r(char="nyx")
    assert c["dim"] == 24 and c["steps"] == 1234


def test_profile_beats_per_char():
    c = r(char="nyx", profile="fast")
    assert c["dim"] == 8 and c["steps"] == 800


def test_cli_beats_profile():
    assert r(char="nyx", profile="fast", overrides={"dim": "64"})["dim"] == 64


def test_unknown_key_rejected():
    with pytest.raises(SystemExit):
        r(char="aria", overrides={"bogus": "1"})


def test_unknown_profile_rejected():
    with pytest.raises(SystemExit):
        r(char="aria", profile="does_not_exist")


def test_real_train_toml_defaults_match_history():
    """The shipped train.toml [defaults] must reproduce the values train_lora.ps1 used to hardcode."""
    c = T.resolve(char="aria")          # loads the real train.toml
    assert c["dim"] == 16 and c["alpha"] == 8 and c["steps"] == 1500
    assert c["epochs"] == 4 and c["batch"] == 2 and c["optimizer"] == "prodigy"
    assert c["d_coef"] == 1.0 and c["resolution"] == 1024 and c["seed"] == 42
    assert c["lr_scheduler"] == "cosine" and c["min_snr_gamma"] == "5"
    assert c["mixed_precision"] == "bf16" and c["save_precision"] == "bf16"
    assert c["min_bucket_reso"] == 768 and c["max_bucket_reso"] == 1280
    assert c["num_cpu_threads"] == 8 and c["min_images"] == 12
    assert c["lr"] == "" and c["train_text_encoder"] is False

"""Pure command-building for `dev train` — the logic that was hardest to verify in PowerShell."""
from devtools.train import cmd

# A resolved-config shape (what train_config.resolve returns), mirroring train.toml [defaults].
CFG = {
    "__char__": "aria",
    "dim": 16, "alpha": 8, "network_module": "networks.lora",
    "optimizer": "prodigy", "d_coef": 1.0, "lr": "", "unet_lr": "", "text_encoder_lr": "",
    "steps": 1500, "epochs": 4, "batch": 2,
    "lr_scheduler": "cosine", "min_snr_gamma": "5", "seed": 42, "save_every_n_epochs": 1,
    "resolution": 1024, "min_bucket_reso": 768, "max_bucket_reso": 1280, "bucket_reso_steps": 64,
    "mixed_precision": "bf16", "save_precision": "bf16", "num_cpu_threads": 8,
    "cache_latents": True, "cache_latents_to_disk": True, "sdpa": True, "no_half_vae": True,
    "gradient_checkpointing": True, "train_text_encoder": False,
}


def test_derive_repeats_hits_target():
    repeats, actual, dev = cmd.derive_repeats(30, 1500, 4, 2)
    assert repeats == 25 and actual == 1500 and dev == 0.0


def test_derive_repeats_floor_one():
    repeats, _, _ = cmd.derive_repeats(10_000, 100, 4, 2)
    assert repeats == 1


def test_single_subset_toml_forward_slashes_and_repeats():
    toml = cmd.render_single_subset_toml(CFG, "C:\\ComfyUI\\output\\dataset\\aria", 25)
    assert 'image_dir = "C:/ComfyUI/output/dataset/aria"' in toml
    assert "num_repeats = 25" in toml
    assert "keep_tokens = 1" in toml
    assert "resolution = 1024" in toml


def test_optimizer_prodigy_default():
    args = cmd.optimizer_args(CFG)
    assert "--optimizer_type" in args and "prodigy" in args
    assert "d_coef=1.0" in args
    # prodigy LR is forced to 1.0
    assert args[args.index("--learning_rate") + 1] == "1.0"


def test_optimizer_adamw_respects_lr_override():
    cfg = {**CFG, "optimizer": "adamw", "lr": "4e-4", "unet_lr": "4e-4", "text_encoder_lr": "5e-5"}
    args = cmd.optimizer_args(cfg)
    assert "AdamW" in args
    assert args[args.index("--learning_rate") + 1] == "4e-4"
    assert args[args.index("--text_encoder_lr") + 1] == "5e-5"


def test_build_accelerate_args_shape():
    args = cmd.build_accelerate_args(
        CFG, base="/m/base.safetensors", out_dir="/m/loras", output_name="aria_v1",
        dataset_cfg="/c/aria.toml", trainer_script="sdxl_train_network.py")
    assert args[0] == "launch"
    assert "sdxl_train_network.py" in args
    assert args[args.index("--output_name") + 1] == "aria_v1"
    assert args[args.index("--network_dim") + 1] == "16"
    # TE off by default -> unet-only
    assert "--network_train_unet_only" in args
    # safety toggles present
    for flag in ("--gradient_checkpointing", "--cache_latents", "--sdpa", "--no_half_vae"):
        assert flag in args


def test_build_accelerate_args_train_te_drops_unet_only():
    cfg = {**CFG, "train_text_encoder": True}
    args = cmd.build_accelerate_args(
        cfg, base="b", out_dir="o", output_name="x_v1", dataset_cfg="c", trainer_script="s.py")
    assert "--network_train_unet_only" not in args

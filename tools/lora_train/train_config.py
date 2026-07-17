"""Resolve LoRA training parameters by layering, and print the result as JSON.

`dev train` (devtools/train) resolves params through this with `--char`, an optional `--profile`,
and any explicitly-set CLI flags as `--set key=value` overrides. Precedence (highest wins):

    explicit CLI override  >  -Profile preset  >  [train.<char>]  >  [defaults]

Why profile beats per-char: `-Profile` is a deliberate runtime choice ("give me a quick `fast` run of
nyx"), so it should win over nyx's persisted defaults; only an explicit flag beats it.

The whole point: tune training without editing code. Defaults live in train.toml [defaults]; presets in
[profiles.<name>]; per-character persistent overrides in [train.<char>].

Usage:
    python train_config.py --char aria
    python train_config.py --char nyx --profile complex --set lr=4e-4 --set dim=32
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
TRAIN_TOML = HERE / "train.toml"


def _b(v):
    """Coerce a value to bool, accepting TOML booleans and CLI strings."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# Every tunable param + how to coerce a (possibly string) value. Keys not here are rejected, which is
# how a typo in train.toml or a bad --set is caught instead of silently ignored.
SCHEMA = {
    "dim": int, "alpha": int, "optimizer": str, "d_coef": float,
    "steps": int, "epochs": int, "batch": int, "min_images": int,
    # LR fields are strings; empty "" means "use the optimizer's own built-in default".
    "lr": str, "unet_lr": str, "text_encoder_lr": str,
    "lr_scheduler": str, "min_snr_gamma": str, "seed": int, "resolution": int,
    "min_bucket_reso": int, "max_bucket_reso": int, "bucket_reso_steps": int,
    "mixed_precision": str, "save_precision": str, "save_every_n_epochs": int,
    "num_cpu_threads": int, "network_module": str,
    # Blackwell / 16 GB safety toggles — exposed but normally left on (see train.toml comments).
    "cache_latents": _b, "cache_latents_to_disk": _b, "sdpa": _b,
    "no_half_vae": _b, "gradient_checkpointing": _b, "train_text_encoder": _b,
}


def _coerce(key, val):
    fn = SCHEMA.get(key)
    if fn is None:
        raise SystemExit(f"unknown training param {key!r} — not in the train.toml schema")
    try:
        return fn(val)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"bad value for {key!r}: {val!r} ({e})")


def load_doc():
    if not TRAIN_TOML.exists():
        raise SystemExit(f"train.toml not found at {TRAIN_TOML}")
    return tomllib.loads(TRAIN_TOML.read_text(encoding="utf-8"))


def resolve(char=None, profile=None, overrides=None, doc=None):
    """Apply the precedence chain and return the fully-resolved param dict."""
    doc = load_doc() if doc is None else doc
    out = {}
    for k, v in doc.get("defaults", {}).items():       # base layer (also validates default keys)
        out[k] = _coerce(k, v)
    if char and char in doc.get("train", {}):           # per-character persistent overrides
        for k, v in doc["train"][char].items():
            out[k] = _coerce(k, v)
    if profile:                                         # named preset
        presets = doc.get("profiles", {})
        if profile not in presets:
            raise SystemExit(f"unknown profile {profile!r}; have {sorted(presets)}")
        for k, v in presets[profile].items():
            out[k] = _coerce(k, v)
    for k, v in (overrides or {}).items():              # explicit CLI flags (highest)
        out[k] = _coerce(k, v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Resolve LoRA training params from train.toml.")
    ap.add_argument("--char")
    ap.add_argument("--profile")
    ap.add_argument("--set", action="append", default=[], metavar="key=value",
                    help="explicit override (repeatable); highest precedence")
    args = ap.parse_args()
    overrides = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v
    print(json.dumps(resolve(args.char, args.profile, overrides), indent=2))


if __name__ == "__main__":
    main()

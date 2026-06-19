from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "user/default/workflows/MainGraphv10.json"
OUT = ROOT / "user/default/workflows"

SEED = 1234567890           # shared fixed seed across ALL tiers (comparison)
CKPT = "oneObsession_v19Atypical.safetensors"
VAE = "sdxl_vae_f16_fix.safetensors"
UPSCALE = "4x-AnimeSharp.pth"
CN_DEPTH = "SDXL\\controlnet-depth-sdxl-1.0\\diffusion_pytorch_model.safetensors"
CN_UNION = "SDXL\\controlnet-union-sdxl-1.0\\diffusion_pytorch_model_promax.safetensors"

# base sampler config — IDENTICAL in every tier (this is what made base best)
BASE_SAMPLER = "euler_ancestral"
BASE_SCHED = "normal"
BASE_STEPS = 28
BASE_CFG = 5

POS = ("masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
       "highly detailed, 1girl, solo, long hair, detailed face, beautiful detailed eyes, "
       "looking at viewer, upper body, soft natural lighting, simple background")
NEG = ("worst quality, low quality, lowres, jpeg artifacts, blurry, bad anatomy, bad hands, "
       "bad proportions, missing fingers, extra digits, fewer digits, fused fingers, "
       "extra limbs, missing limbs, malformed limbs, deformed, disfigured, mutated, "
       "text, watermark, signature, username, artist name, "
       "embedding:negativeXL_D, embedding:BadDigitalHandsNeg, embedding:unaestheticXLv31")
HAND_POS = "detailed hand, perfect hand anatomy, five fingers, correct number of fingers, natural hand pose"
FACE_POS = "detailed face, beautiful detailed eyes, symmetrical eyes, sharp focus, detailed skin texture, natural lips"
NOTE_C, NOTE_BG = "#432", "#322"

# IL_DatasetEdit ROSTER — the character list now lives in the data file `characters.toml`
# (next to this module), so adding a character is an edit-data-and-rebuild step, NOT a code change.
# See characters.toml for the per-field docs (id / outfit / like / hero_seed / prune / trigger).
# Every entry also gets a roster.json line (name/trigger/id/outfit/prune) for the trainer.
def _load_characters():
    """Load characters.toml as an ordered {name: spec} dict (table order preserved)."""
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:  # py3.10
        import tomli as tomllib
    path = Path(__file__).resolve().parent / "characters.toml"
    if not path.exists():
        raise RuntimeError(f"characters.toml not found at {path} — the IL_DatasetEdit roster lives there")
    return tomllib.loads(path.read_text(encoding="utf-8"))


CHARACTERS = _load_characters()
# Suffix that turns the identity tags into a clean FULL-BODY Stage-1 hero (the edit's identity AND outfit
# anchor) -- full body so the whole signature outfit is captured for Qwen to propagate; if a character
# has no lower-body outfit to lock you can shorten this to a portrait for a larger face in the preview.
REF_SUFFIX = (", full body, standing, plain grey background, simple background, looking at viewer, "
              "neutral expression")

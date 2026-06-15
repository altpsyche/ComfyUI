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
BASE_STEPS = 30
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

# IL_Dataset ROSTER — one entry per character you want to train. build_il_graphs.py emits an
# IL_Dataset_<name> workflow per entry (open it in ComfyUI, generate -> output/dataset/<name>/),
# and train_lora.ps1 -Char <name> / train_all.ps1 train them. No per-character file editing.
#   id           identity tags only (face/hair/eyes/body) — what the LoRA bakes into the trigger.
#   outfit       clothes, kept separate from identity.
#   vary_outfit  True -> __outfit__ wildcard (swappable-outfit LoRA); False -> fixed signature outfit.
#   prune        (optional) exact tags train_lora bakes into the trigger; "" = leave identity promptable.
#   base         (optional) known danbooru character tag prepended to the prompt. When set, the tag
#                carries a consistent face, so the hero+IPAdapter scaffold is auto-OFF (pure-text path).
#                Paste the tag RAW (parens and all, e.g. "ganyu (genshin impact)") — build_dataset
#                escapes the parens so CLIP doesn't read them as prompt weights. "" (default) =
#                original face via the in-graph hero + light IPAdapter. (OLD route only.)
#   hero         (optional) filename in ComfyUI/input/ for the HERO OVERRIDE LoadImage in
#                IL_DatasetEdit_<name> (use your own image instead of the in-graph Stage-1 render).
#                "" -> defaults to "<name>_hero.png". Not needed normally -- Stage 1 makes the hero.
#   hero_graph   (optional) True also emits the OLD IL_Dataset_<name> (hero+IPAdapter / base) graph.
#                Default False: only the recommended IL_DatasetEdit_<name> (Qwen-Image-Edit) is emitted.
# Every entry always gets a roster.json line (name/trigger/prune) for the trainer, regardless of route.
CHARACTERS = {
    # DEFAULT route: Qwen-Image-Edit. Render aria_hero.png into ComfyUI/input/, open IL_DatasetEdit_aria.
    "aria": {
        "id": "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles",
        "prune": "",
        "hero": "aria_hero.png",
    },
    # Minimal entry: identity only; trigger defaults to kaelchar, hero defaults to kael_hero.png.
    "kael": {
        "id": "1boy, solo, (tousled black hair:1.1), (sharp blue eyes:1.1)",
        "prune": "",
    },
    # OLD hero+IPAdapter route too (hero_graph=True) + the danbooru base path (base-only, IPAdapter OFF).
    # Swap "base" to ANY Danbooru-2024 character your checkpoint renders reliably before generating.
    "nyx": {
        "id": "1girl, solo",                  # keep id minimal; the base tag supplies the face
        "outfit": "casual hoodie, jeans",
        "vary_outfit": False,
        "prune": "",
        "hero_graph": True,
        "base": "ganyu (genshin impact)",
    },
}
# Suffix that turns the identity tags into a clean hero portrait (the IPAdapter face source).
REF_SUFFIX = (", upper body, plain grey background, simple background, looking at viewer, "
              "neutral expression, character portrait")

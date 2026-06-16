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

# IL_DatasetEdit ROSTER — one entry per character you want to train. build_il_graphs.py emits an
# IL_DatasetEdit_<name> (Qwen-Image-Edit, self-contained two-stage) workflow per entry: open it in
# ComfyUI, generate -> output/dataset/<name>/, then train_lora.ps1 -Char <name> / train_all.ps1.
# No per-character file editing.
#   id      identity tags only (face/hair/eyes/body). This is the STAGE-1 hero prompt (rendered in
#           your checkpoint, then re-posed by Qwen-Edit). Weight the face-defining tags, e.g.
#           "(green eyes:1.1)". No outfit here.
#   outfit  (optional) signature clothes, kept separate from identity; appended to the Stage-1 hero
#           prompt so the hero wears them. Leave "" to let the checkpoint pick.
#   prune   (optional) exact tags train_lora bakes into the trigger; "" = leave identity promptable.
# Every entry also gets a roster.json line (name/trigger/prune) for the trainer.
CHARACTERS = {
    # Tennis player. Teal complements warm auburn hair; white flatters fair/freckled skin; nods to green eyes.
    # Locked outfit: `prune` folds the garment tags into the trigger so it always renders the same kit.
    "aria": {
        "id": "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles",
        "outfit": "tennis uniform, teal and white tennis dress, white visor, white wristbands, white shoes",
        "prune": "tennis uniform, tennis dress, dress, visor, wristband, shoes",
    },
    # Basketball player. Orange is the complement of blue eyes (pops) and vivid against black hair.
    "kael": {
        "id": "1boy, solo, (tousled black hair:1.1), (sharp blue eyes:1.1)",
        "outfit": "basketball uniform, orange and white basketball jersey, orange basketball shorts, white headband, basketball shoes",
        "prune": "basketball uniform, basketball, jersey, shorts, headband, shoes",
    },
    # Superhero. Deep violet echoes the eyes, silver echoes the hair, both striking on pale skin.
    "nyx": {
        "id": "1girl, solo, (silver bob hair:1.1), (violet eyes:1.1)",
        "outfit": "superhero costume, deep violet bodysuit, silver accents, silver belt, knee boots, purple cape",
        "prune": "superhero, bodysuit, cape, belt, boots, gloves",
    },
}
# Suffix that turns the identity tags into a clean FULL-BODY Stage-1 hero (the edit's identity AND outfit
# anchor) -- full body so the whole signature outfit is captured for Qwen to propagate; if a character
# has no lower-body outfit to lock you can shorten this to a portrait for a larger face in the preview.
REF_SUFFIX = (", full body, standing, plain grey background, simple background, looking at viewer, "
              "neutral expression")

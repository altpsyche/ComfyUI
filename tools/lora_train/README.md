# Train a character-consistency LoRA

End-to-end: generate a dataset (ComfyUI), set up kohya sd-scripts (one-time), caption, train,
then load the LoRA in any IL workflow's LoRA bank. Repeat per character (2 separate LoRAs).

## 1. Generate the dataset  (ComfyUI — manual)

1. Edit `CHAR` in `tools/il_graphs/config.py` to character A's weighted identity tags; run
   `python tools/build_il_graphs.py`.
2. Open **IL_Dataset** in ComfyUI. Keep **Hero Seed** fixed; queue once and check the hero
   portrait (PREVIEW the "Hero decode") — that face is what gets locked. Reroll Hero Seed until
   you like it, then leave it fixed.
3. Reroll the **Gen Seed** and queue ~15 times (batch of 4 each) → ~60 shots land in
   `output/dataset/char/`.
4. **Curate** down to the best **25–40** on-model, pose/angle-varied images. Delete
   melted/merged/off-color/duplicate faces. Move them to `output/dataset/charA/`.
5. Repeat 1–4 for character B into `output/dataset/charB/` (re-edit `CHAR`, regenerate).

## 2. Set up the trainer venv  (one-time, Blackwell-ready)

sd-scripts is vendored in-repo as a submodule at **`tools/sd-scripts`** (a fresh `setup.bat`
initializes it in phase [2/6]). The trainer venv lives at `tools/lora_train/.venv` (gitignored,
kept out of the submodule) and is provisioned by setup the right way:

```powershell
setup.bat --with-trainer
```

That runs `scripts/install_trainer.ps1`: creates the venv (uv, Python 3.11 — the ML stack needs
≤3.12), installs **torch cu128** (the 5080 / sm_120 needs CUDA 12.8+; default kohya torch won't
run), the sd-scripts requirements, the WD14-tagger deps (onnx/onnxruntime), and `prodigyopt`.
Idempotent. To (re)provision the trainer alone without a full setup:
`powershell -ExecutionPolicy Bypass -File scripts\install_trainer.ps1`.

Sanity-check the venv (GPU compute + sd-scripts + tagger imports):
`tools\lora_train\.venv\Scripts\python.exe tools\lora_train\verify_env.py`

## 3. Caption  (venv active, run from the submodule dir)

```powershell
cd C:/Users/vsiva/dev/ComfyUI/tools/sd-scripts
# WD14 booru tags (matches Illustrious' training distribution):
python finetune/tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-v1-4-convnextv2-tagger-v2 `
  --batch_size 4 "C:/Users/vsiva/dev/ComfyUI/output/dataset/charA"
# Then prepend a unique trigger + bake identity tags:
python "C:/Users/vsiva/dev/ComfyUI/tools/lora_train/prep_captions.py" `
  "C:/Users/vsiva/dev/ComfyUI/output/dataset/charA" --trigger ariacharA `
  --prune "auburn hair,long hair,wavy hair,green eyes,freckles"
```

Keep variable tags (pose/expression/framing/background); prune the persistent identity tags so
they fold into the trigger. Pick a *rare* trigger string (e.g. `ariacharA`, not `aria`).

## 4. Train  (venv active)

```powershell
.\tools\lora_train\.venv\Scripts\Activate.ps1
& "C:/Users/vsiva/dev/ComfyUI/tools/lora_train/train_charA.ps1"   # cd's into tools/sd-scripts itself
```

Settings (in the script + `dataset_charA.toml`): LoRA dim 16 / alpha 8, Prodigy lr 1.0 cosine,
min_snr_gamma 5, resolution 1024 + bucketing 768–1280, bf16, batch 2, gradient checkpointing,
cache_latents_to_disk, `--sdpa` (no xformers — avoids Blackwell wheel pain), unet-only (16 GB-safe).
~1200–1500 steps. Saves `charA_aria_v1.safetensors` (+ one per epoch) to `models/loras/`.
The script sets `PYTHONUTF8=1` — sd-scripts prints unicode progress that crashes a cp1252 console.

Copy `dataset_charA.toml` → `dataset_charB.toml` (change `image_dir`) and `train_charA.ps1`
→ `train_charB.ps1` (change `$DATACFG`/`$NAME` + the trigger in captioning) for character B.

## 5. Use it  (any IL workflow)

Open IL_1_Base (or any tier). In the **LoRA bank** node: toggle `charA_aria_v1` ON, strength
~0.75, and put the trigger (`ariacharA`) in the Positive prompt. Identity flows through base +
every detail pass automatically.

**Pick the best epoch:** XY-plot strength {0.5, 0.75, 0.9} × a few seeds with the trigger; choose
the epoch/strength that holds identity without frying style. If identity is weak, raise strength
or train with the text encoder (drop `--network_train_unet_only`); if it overfits the dataset
poses, lower num_repeats/epochs.

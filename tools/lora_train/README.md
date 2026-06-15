# Train character-consistency LoRAs (one or many)

Define your characters once in a **roster**, generate a dataset per character (ComfyUI), set up
kohya sd-scripts (one-time), then **one command per character** (or one for the whole roster)
captions + trains. Load each LoRA in any IL workflow's LoRA bank. Fully parameterized — no
per-character file copies, no manual file moving.

## 1. Define the roster + generate datasets  (ComfyUI)

1. Edit the **`CHARACTERS`** roster in `tools/il_graphs/config.py` — one entry per character
   (`id` = identity tags, `outfit`, optional `vary_outfit`/`prune`). Run `python tools/build_il_graphs.py`.
   This emits one **`IL_Dataset_<name>`** workflow per character (e.g. `IL_Dataset_aria`,
   `IL_Dataset_kael`) and a `roster.json` the trainer reads.
2. Open **`IL_Dataset_<name>`** in ComfyUI. Keep **Hero Seed** fixed; queue once and check the hero
   portrait — that face gets locked. Reroll Hero Seed until you like it, then leave it fixed.
3. Reroll the **Gen Seed** and queue ~15× (batch of 4) → ~60 shots land **straight in
   `output/dataset/<name>/`** (each character its own folder — no collisions, no moving).
4. **Curate in place:** delete the off-model / melted / duplicate shots, keeping the best
   **25–40**. Repeat 2–4 for each character's graph.

**Outfit:** by default every shot wears the entry's `outfit` → the LoRA reproduces that *signature*
outfit. For a **swappable-outfit** LoRA, set `"vary_outfit": True` on that roster entry: the dataset
varies clothes via the `__outfit__` wildcard so the LoRA learns the face/body, not the clothes — and
**don't** prune outfit tags at train time so they stay promptable.

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

## 3. Caption + train  (one command)

```powershell
.\tools\lora_train\train_lora.ps1 -Char aria      # one character
.\tools\lora_train\train_all.ps1                  # every roster character with a curated dataset
```

`train_lora.ps1` (portable — derives every path from its own location) does the rest:
1. validates `output/dataset/<Char>/` has enough images,
2. **auto-captions** if needed — WD14 booru tagger, then prepends the trigger and prunes any exact
   identity tags (trigger + prune come from the roster; pose/expression/framing tags stay),
3. **derives `num_repeats`** from a target step count, so 15 or 50 images both land near target,
4. writes the dataset config and trains.

`train_all.ps1` reads `roster.json` and runs `train_lora.ps1` for each character that has a curated
dataset (skipping ones you haven't generated yet). Extra args pass through, e.g. `train_all.ps1 -Steps 2000`.

Defaults: trigger `<Char>char` + prune from the roster, base `oneObsession`, dim 16 / alpha 8,
Prodigy lr 1.0 cosine, min_snr 5, res 1024 + bucketing, bf16, batch 2, `--sdpa`, unet-only, ~1500
steps. Override any: `-Trigger`, `-Prune`, `-Base <ckpt>`, `-Dim`, `-Alpha`, `-Steps`, `-Epochs`,
`-Batch`, `-TrainTextEncoder`, `-SkipCaption`. Output → `models/loras/<Char>_v1.safetensors`.

## 4. Use it  (any IL workflow)

Open IL_1_Base (or any tier). In the **LoRA bank** node: toggle `aria_v1` ON, strength ~0.75, and
put the trigger (`ariachar`) in the Positive prompt. Identity flows through base + every detail
pass automatically.

**Pick the best epoch:** XY-plot strength {0.5, 0.75, 0.9} × a few seeds with the trigger; choose
the epoch/strength that holds identity without frying style. If identity is weak, raise strength
or train with the text encoder (drop `--network_train_unet_only`); if it overfits the dataset
poses, lower num_repeats/epochs.

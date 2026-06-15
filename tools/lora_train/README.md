# Train a character-consistency LoRA

End-to-end: generate a dataset (ComfyUI), set up kohya sd-scripts (one-time), then **one command**
captions + trains. Load the result in any IL workflow's LoRA bank. One LoRA per character; the
flow is fully parameterized — no per-character file copies, no manual file moving.

## 1. Generate the dataset  (ComfyUI)

1. Open **IL_Dataset**. Set the character: edit the **Character identity** prompt and the
   **SaveImage prefix** to `dataset/<name>` (e.g. `dataset/aria`) — both in the UI, no regen.
   (Or set `CHAR_NAME`/`CHAR` in `tools/il_graphs/config.py` and re-run `build_il_graphs.py`.)
2. Keep **Hero Seed** fixed; queue once and check the hero portrait — that face gets locked.
   Reroll Hero Seed until you like it, then leave it fixed.
3. Reroll the **Gen Seed** and queue ~15× (batch of 4) → ~60 shots land **straight in
   `output/dataset/<name>/`** (each character has its own folder — no collisions, no moving).
4. **Curate in place:** delete the off-model / melted / duplicate shots, keeping the best
   **25–40**. That's it — they're already where the trainer expects them.

For a second character, change the prompt + SaveImage prefix to `dataset/<other>` and repeat.

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
.\tools\lora_train\train_lora.ps1 -Char aria -Prune "auburn hair,long hair,green eyes,freckles"
```

That single script (portable — derives every path from its own location) does the rest:
1. validates `output/dataset/aria/` has enough images,
2. **auto-captions** if needed — WD14 booru tagger, then prepends the trigger and prunes the exact
   identity tags you name (they bake into the trigger; pose/expression/framing tags stay),
3. **derives `num_repeats`** from a target step count, so 15 or 50 images both land near target,
4. writes the dataset config and trains.

Defaults: trigger `<Char>char`, base `oneObsession`, dim 16 / alpha 8, Prodigy lr 1.0 cosine,
min_snr 5, res 1024 + bucketing, bf16, batch 2, `--sdpa`, unet-only, ~1500 steps. Override any:
`-Trigger`, `-Base <ckpt>`, `-Dim`, `-Alpha`, `-Steps`, `-Epochs`, `-Batch`, `-TrainTextEncoder`,
`-SkipCaption`. Output → `models/loras/<Char>_v1.safetensors` (+ one per epoch). A second character
is just another call: `train_lora.ps1 -Char kael -Prune "..."` — same script, no copies, no moving.

## 4. Use it  (any IL workflow)

Open IL_1_Base (or any tier). In the **LoRA bank** node: toggle `aria_v1` ON, strength ~0.75, and
put the trigger (`ariachar`) in the Positive prompt. Identity flows through base + every detail
pass automatically.

**Pick the best epoch:** XY-plot strength {0.5, 0.75, 0.9} × a few seeds with the trigger; choose
the epoch/strength that holds identity without frying style. If identity is weak, raise strength
or train with the text encoder (drop `--network_train_unet_only`); if it overfits the dataset
poses, lower num_repeats/epochs.

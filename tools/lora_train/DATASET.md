# Dataset engine — `IL_DatasetEdit_<name>` (Qwen-Image-Edit)

> How the training images are made. Concepts + the loop: [README.md](README.md) ·
> add a character: [ADD_CHARACTER.md](ADD_CHARACTER.md) · steer variety: [WILDCARDS.md](WILDCARDS.md) ·
> traps: [GOTCHAS.md](GOTCHAS.md).

`IL_DatasetEdit_<name>` is the 2026 way to bootstrap a dataset for a **fully-original** character —
one that doesn't resemble any danbooru character, so a text tag can't carry the face. One
**self-contained, two-stage** graph per roster entry:

- **STAGE 1** renders ONE hero from the character's `id` tags **in your own SDXL checkpoint** (you
  reroll a fixed **Hero Seed** and pick the face) — so a brand-new character needs **no input image**.
- **STAGE 2** lets an **image-edit model re-pose that whole hero** (face + hair + body + outfit) into
  new angles/poses/scenes, holding identity *and* your art style. Edit models only change what the
  instruction asks and are conditioned on the input image, so every frame stays on-style — no realism
  drift. (An optional IL img2img low-denoise re-skin can follow, but isn't needed in practice.)

The dataset only needs *recognizably the same person*; curation drops outliers and the trained LoRA
averages the rest into the final exact face. Curate → `train_lora.ps1`.

## One-time setup

The model is **Qwen-Image-Edit-2511**, run as a quantized **GGUF** to fit a 16 GB GPU. It needs the
**ComfyUI-GGUF** node (a submodule added by `setup.bat`; if missing:
`git submodule update --init custom_nodes/ComfyUI-GGUF` then install its `requirements.txt` into the
ComfyUI venv).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1
#   -Quant Q4_K_M    # smaller/faster, lower quality   (default Q5_K_M)
#   -SkipAnglesLora  # skip the camera-angles LoRA
```
Idempotently downloads (~23 GB total) into the right `models/` subfolders:

| File | → folder | Role |
|---|---|---|
| `qwen-image-edit-2511-Q5_K_M.gguf` (~15 GB) | `models/unet/` | the edit diffusion model (GGUF) |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` (~9 GB) | `models/text_encoders/` | Qwen 2.5-VL text/vision encoder |
| `qwen_image_vae.safetensors` (~250 MB) | `models/vae/` | Qwen-Image VAE |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` (~850 MB) | `models/loras/` | 4-step distill (makes 16 GB practical) |
| `qwen-image-edit-2511-multiple-angles-lora.safetensors` (~295 MB) | `models/loras/` | drives camera-angle variety |

> **Quant guide (16 GB):** `Q5_K_M` is the sweet spot (usable quality, encoder offloaded to RAM).
> `Q4_K_M` on OOM / for speed (noticeably weaker). `Q6_K` only with VRAM headroom. Re-run the
> installer with a different `-Quant` to swap — it skips files already present.

Then `python tools/build_il_graphs.py` emits one `IL_DatasetEdit_<name>` per roster character with the
Stage-1 prompt (the `id`), Stage-2 model, and `dataset/<name>/<name>` save prefix all pre-wired.

## Graph anatomy

| Group | Does |
|---|---|
| **STAGE 1 — Hero (Illustrious)** | `CheckpointLoaderSimple` + `CLIPSetLastLayer −2` + the `id` prompt + `KSampler` (euler_a/normal/30/cfg 5, fixed **Hero Seed**) → `VAEDecode` → **HERO preview**. The single hero, in your style, no input image. |
| **STAGE 2 — Qwen-Edit model** | `UnetLoaderGGUF` (Q5) → `LoraLoaderModelOnly` ×2 (Lightning 1.0, multiple-angles 0.8) → `ModelSamplingAuraFlow` (shift 3.1) → `CFGNorm` (1.0). The official 2511 model-patch chain. |
| **Encoders + scale** | `CLIPLoader` (qwen2.5-vl-7b, type `qwen_image`), `VAELoader` (qwen_image_vae), `FluxKontextImageScale` (scales the hero to the model's pixel budget). |
| **Instruction + encode** | **`Edit instruction`** (`ImpactWildcardProcessor`, **mode `populate`** — see below) → `TextEncodeQwenImageEditPlus` (positive: scaled hero + instruction; negative: empty). Each conditioning passes a `FluxKontextMultiReferenceLatentMethod` node (kept **ON** — required for the repackaged GGUF). `VAEEncode` makes the init latent. |
| **Edit + decode** | `KSampler` (Lightning: **6 steps / cfg 1.0 / euler / simple / denoise 1.0**) → `VAEDecode`. |
| **Save** | `SaveImage` prefix `dataset/<name>/<name>` → `output/dataset/<name>/`. |

## Step-by-step

1. **Open `IL_DatasetEdit_<name>`** (re-open after any regenerate — ComfyUI caches loaded graphs).
   Everything is pre-wired from the roster.
2. **Stage 1 — pick the face.** Reroll the **Hero Seed** and watch **HERO preview** until you like the
   face. Leave Hero Seed **fixed** on that value — that image is now the identity anchor. (Pin it in
   `characters.toml` as `hero_seed = <value>` so `like` variants reuse it.)
3. **Stage 2 — confirm variety.** The **Edit instruction** node is `mode: populate`, seed control
   **randomize**. Each queue, the bottom box shows the resolved prompt and re-rolls a new
   framing/angle/pose/expression/background/lighting.
4. Set **batch count** beside Queue to ~40 and **Queue once** → ~40 varied frames stream into
   `output/dataset/<name>/`. (One edit per queue — use a higher batch count than a batched txt2img.)
5. **Curate:** delete melted/off-model/duplicate frames **in place**. Keep the best **25–40** (min
   **12**), varied in pose/angle/scene. Then train: `train_lora.ps1 -Char <name>` (or `train_all.ps1`).

## The edit instruction (driving variety)

The positive prompt is produced by `ImpactWildcardProcessor`, shipping as:
```
same character, identical face and hair and outfit, keep the same art style,
__angle__, __pose__, __expression__, __framing__, __background__, __lighting__
```
- **Keep `__angle__/__pose__` leading.** A "lead with the change" rewrite
  (`Change the shot to __framing__ … Re-pose to __pose__ …`) was **reverted**: at 6 steps Qwen is
  conservative, so framing/scene-first makes it spend the budget on zoom/background/lighting and move
  the **pose** much less. The identity-preamble comma-list (angle/pose first) gives the most pose
  variety. **Append** new axes after pose; never reorder pose behind them.
- **`mode: populate`, wildcards in `populated_text`.** In the UI, `populate` re-expands `wildcard_text`
  into the read-only `populated_text` box each queue, so you **see** the resolved prompt and it
  re-rolls. The generator seeds `populated_text` with the wildcard string itself (not a concrete roll),
  so a **headless API POST** — no frontend to populate — still expands in the node backend
  (`doit()` → `process(populated_text, seed)`), keyed on the seed. (Early headless sameness came from a
  *concrete, wildcard-free* default, not from `populate`.)
- The six `__token__`s are Impact-Pack wildcards (one random line each, from
  `custom_nodes/ComfyUI-Impact-Pack/wildcards/*.txt`). Edit those files to steer variety — see
  [WILDCARDS.md](WILDCARDS.md).
- The **multiple-angles LoRA** (strength **0.8**) reinforces camera-angle changes; raise toward 1.0 for
  more push, lower if identity drifts.

If you settle on better defaults (LoRA strengths, steps, instruction), bake them into
`build_dataset_edit()` in [`graphs.py`](../il_graphs/graphs.py) so regenerations keep them
(see [il_graphs/ARCHITECTURE.md](../il_graphs/ARCHITECTURE.md)).

## Tuning dials (live in the graph)

| Symptom | Dial |
|---|---|
| Poses too similar | confirm seed control = randomize and the bottom box re-rolls; keep `__angle__/__pose__` leading; raise multiple-angles LoRA (0.8 → 1.0); add lines to `pose.txt`/`angle.txt` |
| Identity drifts across frames | lower multiple-angles LoRA (0.8 → 0.6); trim the scene axes (`__background__/__lighting__`); use a cleaner hero |
| Soft / low-detail output | raise KSampler steps (6 → 8–10), keep the Lightning LoRA; cfg can stay 1.0 |
| Style drifts from your checkpoint | ensure the hero was rendered in your checkpoint; optionally add an IL img2img re-skin (denoise 0.25–0.35) after |
| Output too zoomed/cropped | the ref auto-scales via `FluxKontextImageScale`; add framing words to `framing.txt` |
| Too slow per frame | per-frame compute is fixed by KSampler steps (6) + the model stack — wildcards add none. A real slowdown is environmental: a cold reload of the ~24 GB stack after regenerate, other GPU apps, or VRAM pressure (the 9 GB encoder offloads to RAM). Time a few frames; use `-Quant Q4_K_M` if the encoder swap dominates |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Node/changes not showing | A loaded graph is cached — **re-open** the workflow after regenerating. |
| Every frame the same pose/angle | First the *prompt* must roll (`mode: populate`, seed = randomize — bottom box changes each queue). If the prompt rolls but the *image* doesn't, that's Qwen being conservative: keep `__angle__/__pose__` leading and raise the multiple-angles LoRA toward 1.0. Headless POSTs need the seed-randomizing runner (`convert_and_run.py`); a raw POST keeps the saved seed. |
| Images for all characters in one folder | Old prefix bug; ensure the SaveImage prefix is `dataset/<name>/<name>` (regenerate). |
| `__pose__` etc. appear literally in the image | Wildcard file missing/misnamed — files go in `custom_nodes/ComfyUI-Impact-Pack/wildcards/`; reload graph. |
| `ImpactWildcardProcessor` missing / red | Impact-Pack not loaded — `setup.bat` / `install_node_reqs.ps1`. |
| `UnetLoaderGGUF` missing / red | ComfyUI-GGUF not loaded — `git submodule update --init custom_nodes/ComfyUI-GGUF` + install its `requirements.txt`. |
| Model not in a dropdown | not downloaded — run `scripts/install_qwen_edit.ps1`; confirm it landed in the listed `models/` subfolder. |
| Edited frame ignores the hero | confirm Stage-1 `Hero decode` feeds **Scale ref**, and that feeds **image1** on both encoders; keep the reference-method nodes ON. |
| Output not anime / off-style | Stage 1 renders in your checkpoint (`CKPT`) — if off, tighten `id` or add an IL img2img re-skin pass. |
| Stage-1 hero looks wrong | tighten the `id` tags (weight face-defining ones); reroll the Hero Seed. |
| OOM / too slow generating | `scripts\install_qwen_edit.ps1 -Quant Q4_K_M`; close other GPU apps (check `nvidia-smi`). |

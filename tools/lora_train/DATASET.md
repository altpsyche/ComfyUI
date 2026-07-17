# Dataset engine — `IL_DatasetEdit_<name>` (Qwen-Image-Edit)

> How the training images are made. Concepts + the loop: [README.md](README.md) ·
> add a character: [ADD_CHARACTER.md](ADD_CHARACTER.md) · steer variety: [WILDCARDS.md](WILDCARDS.md) ·
> traps: [GOTCHAS.md](GOTCHAS.md).

`IL_DatasetEdit_<name>` is the 2026 way to bootstrap a dataset for a **fully-original** character —
one that doesn't resemble any danbooru character, so a text tag can't carry the face. One
**self-contained** graph per roster entry:

- **STAGE 1** renders ONE hero from the character's `id` tags **in your own SDXL checkpoint** (you
  reroll a fixed **Hero Seed** and pick the face) — so a brand-new character needs **no input image**.
- **STAGE 1b** (default ON) face+hand **details that one hero** in your SDXL checkpoint (clean
  identity-only face prompt, denoise ~0.35) BEFORE the edit — so a crisp, on-model character is the
  source every frame inherits. This is **one** detail pass total (the hero), not one per saved frame.
  Toggle `QE_HERO_DETAIL` in `il_graphs/graphs.py`.
- **STAGE 2** lets an **image-edit model re-pose that detailed hero** (face + hair + body + outfit) into
  new angles/poses/scenes, holding identity *and* your art style. Edit models only change what the
  instruction asks and are conditioned on the input image, so every frame stays on-style — no realism
  drift.
- **STAGE 3** (optional, **off** by default) can ALSO re-detail each *edited frame* — enable
  `QE_STAGE3_POLISH` only if Qwen still softens faces despite the detailed hero (belt-and-suspenders, but
  it adds a per-frame SDXL pass, so it's slower).

The dataset only needs *recognizably the same person*; curation drops outliers and the trained LoRA
averages the rest into the final exact face. Curate → `./dev train`.

## One-time setup

The model is **Qwen-Image-Edit-2511**, run as a quantized **GGUF** to fit a 16 GB GPU. It needs the
**ComfyUI-GGUF** node (a submodule added by `./dev setup`; if missing:
`git submodule update --init custom_nodes/ComfyUI-GGUF` then install its `requirements.txt` into the
ComfyUI venv).

Run `./dev …` on Linux/macOS, `dev …` on Windows.

```bash
./dev models install il_graphs
#   --variant quant=Q4_K_M    # smaller/faster, lower quality   (default Q5_K_M)
```
Both the Lightning and multiple-angles LoRAs are always fetched.
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
> installer with a different `--variant quant=…` to swap — it skips files already present.

Then `python tools/build_il_graphs.py` emits one `IL_DatasetEdit_<name>` per roster character with the
Stage-1 prompt (the `id`), Stage-2 model, and `dataset/<name>/<name>` save prefix all pre-wired.

## Graph anatomy

| Group | Does |
|---|---|
| **STAGE 1 — Hero (Illustrious)** | `CheckpointLoaderSimple` + `CLIPSetLastLayer −2` + the `id` prompt + `KSampler` (euler_a/normal/30/cfg 5, fixed **Hero Seed**) → `VAEDecode`. The single hero, in your style, no input image. |
| **STAGE 1b — Hero detail** *(default ON)* | `FaceDetailer` ×2 (face + hand) in the SDXL hero checkpoint — face pass uses a clean identity-only prompt at denoise `QE_HERO_FACE_DENOISE` (0.35). Crisps the hero **once** so every edited frame inherits an on-model face → **HERO preview** shows this. Skip with `QE_HERO_DETAIL=False`. |
| **STAGE 2 — Qwen-Edit model** | `UnetLoaderGGUF` (Q5) → `LoraLoaderModelOnly` ×2 (Lightning 1.0, multiple-angles 0.8) → `ModelSamplingAuraFlow` (shift 3.1) → `CFGNorm` (1.0). The official 2511 model-patch chain. |
| **Encoders + scale** | `CLIPLoader` (qwen2.5-vl-7b, type `qwen_image`), `VAELoader` (qwen_image_vae), `FluxKontextImageScale` (scales the **detailed** hero to the model's pixel budget). |
| **Instruction + encode** | **`Edit instruction`** (`ImpactWildcardProcessor`, **mode `populate`** — see below) → `TextEncodeQwenImageEditPlus` (positive: scaled hero + instruction; negative: empty). Each conditioning passes a `FluxKontextMultiReferenceLatentMethod` node (kept **ON** — required for the repackaged GGUF). `VAEEncode` makes the init latent. |
| **Edit + decode** | `KSampler` (Lightning: **6 steps / cfg 1.0 / euler / simple / denoise 1.0**) → `VAEDecode` → **Save** (no per-frame detail — the hero was detailed up front). |
| **STAGE 3 — polish** *(optional, default OFF)* | `FaceDetailer` ×2 on each *edited frame* — enable `QE_STAGE3_POLISH=True` only if Qwen still softens faces despite the detailed hero. Adds a per-frame SDXL pass (slower). |
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
   *Headless / unattended:* one-time **File → Export (API)** the graph as
   `IL_DatasetEdit_<name>.api.json`, then `python tools/lora_train/gen_dataset.py <name> -n 40` queues
   it N times with fresh seeds (or `--all` for the roster). See [REFERENCE.md](REFERENCE.md).
5. **Curate:** delete melted/off-model/duplicate frames **in place**. Keep the best **25–40** (min
   **12**), varied in pose/angle/scene. Then train: `./dev train <name>` (or `./dev train --all`).

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
- The six `__token__`s are wildcards (one random line each) read from the tracked
  `tools/il_graphs/wildcards/*.txt`. Edit those files to steer variety — see [WILDCARDS.md](WILDCARDS.md).
- The **multiple-angles LoRA** (strength **0.8**) reinforces camera-angle changes; raise toward 1.0 for
  more push, lower if identity drifts.

If you settle on better defaults (LoRA strengths, steps, instruction), bake them into
`build_dataset_edit()` in [`graphs.py`](../il_graphs/graphs.py) so regenerations keep them
(see [il_graphs/ARCHITECTURE.md](../il_graphs/ARCHITECTURE.md)).

## Tuning dials (live in the graph)

| Symptom | Dial |
|---|---|
| Poses too similar | confirm seed control = randomize and the bottom box re-rolls; keep `__angle__/__pose__` leading; raise multiple-angles LoRA (0.8 → 1.0); add lines to `pose.txt`/`angle.txt` |
| Identity drifts across frames | the hero is detailed up front (Stage 1b) — raise `QE_HERO_FACE_DENOISE` (0.35 → 0.5) for a stronger hero face; also lower multiple-angles LoRA (0.8 → 0.6), trim scene axes, use a cleaner hero. Still drifting after the edit? enable `QE_STAGE3_POLISH` |
| Soft / low-detail faces | the detailed hero should carry crisp faces through the edit; if still soft, raise the edit KSampler steps (`QE_EDIT_STEPS` 6 → 8–10), or enable `QE_STAGE3_POLISH` for a per-frame face pass |
| Hero preview itself looks low-detail | that's the pre-detail render if `QE_HERO_DETAIL=False`; turn it on (the preview shows the detailed hero). Reroll feels slow? mute the **Face + Hand Detail** group while picking the seed |
| Style drifts from your checkpoint | ensure the hero is rendered in your checkpoint; Stage 1b re-renders its face there too. The face prompt is identity-only by design |
| Faces over-cooked | lower `QE_HERO_FACE_DENOISE` (0.35 → 0.25) |
| Output too zoomed/cropped | the ref auto-scales via `FluxKontextImageScale`; add framing words to `framing.txt` |
| Too slow per frame | the detail pass is now on the hero ONCE (not per frame), so per-edit cost is just `QE_EDIT_STEPS` (6) + the model stack; a real slowdown is environmental (cold ~24 GB reload, other GPU apps, encoder offload). `./dev models install il_graphs --variant quant=Q4_K_M` if the encoder swap dominates. (Don't enable `QE_STAGE3_POLISH` unless you need it — it re-adds a per-frame pass.) |

## Removable garments (coat-off frames)

To make a garment **removable at inference** (e.g. take the overcoat off), two parts:

1. **Keep it promptable** — list it in the character's `keep` field so it's *not* baked into the trigger
   (see [REFERENCE.md](REFERENCE.md) "Removable garments"). Now the prompt controls it.
2. **Show it both ways in the dataset** *(optional but needed for reliable removal)* — if every frame
   wears the coat, the LoRA still learns "trigger ⇒ coat". Generate a handful of **coat-off** frames so
   it learns the character without it:
   - quickest: temporarily drop the garment from the character's `outfit`, regenerate ~10 frames, and
     curate them into the same `output/dataset/<name>/`; or
   - per-frame: append an explicit removal to the edit instruction (e.g. `, remove the coat`) for some
     queues — Qwen-Edit can take layers off. Verify it removes cleanly before committing many frames.

A 70/30 with-coat / without-coat split is plenty for clean on/off control.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Node/changes not showing | A loaded graph is cached — **re-open** the workflow after regenerating. |
| Every frame the same pose/angle | First the *prompt* must roll (`mode: populate`, seed = randomize — bottom box changes each queue). If the prompt rolls but the *image* doesn't, that's Qwen being conservative: keep `__angle__/__pose__` leading and raise the multiple-angles LoRA toward 1.0. Headless? use `gen_dataset.py` (it bumps seeds per POST); a raw POST keeps the saved seed. |
| Images for all characters in one folder | Old prefix bug; ensure the SaveImage prefix is `dataset/<name>/<name>` (regenerate). |
| `__pose__` etc. appear literally in the image | Wildcard file missing/misnamed — files go in `custom_nodes/ComfyUI-Impact-Pack/wildcards/`; reload graph. |
| `ImpactWildcardProcessor` missing / red | Impact-Pack not loaded — re-run `./dev setup`. |
| `UnetLoaderGGUF` missing / red | ComfyUI-GGUF not loaded — `git submodule update --init custom_nodes/ComfyUI-GGUF` + install its `requirements.txt`. |
| Model not in a dropdown | not downloaded — run `./dev models install il_graphs`; confirm it landed in the listed `models/` subfolder. |
| Edited frame ignores the hero | confirm Stage-1 `Hero decode` feeds **Scale ref**, and that feeds **image1** on both encoders; keep the reference-method nodes ON. |
| Output not anime / off-style | Stage 1 renders in your checkpoint (`CKPT`) — if off, tighten `id` or add an IL img2img re-skin pass. |
| Stage-1 hero looks wrong | tighten the `id` tags (weight face-defining ones); reroll the Hero Seed. |
| OOM / too slow generating | `./dev models install il_graphs --variant quant=Q4_K_M`; close other GPU apps (check `nvidia-smi`). |

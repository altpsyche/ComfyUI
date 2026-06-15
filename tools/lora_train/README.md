# Character LoRA training — complete guide (A → Z)

Train one LoRA per character so that character renders **identically every time** in any IL_*
workflow. Text prompts and IPAdapter both drift; a trained LoRA is the only thing that locks an
exact identity. This kit takes you from "no images" to "trained LoRA loaded in the LoRA bank",
fully scripted — no per-character file copies, no manual file moving.

## Contents
1. [Mental model](#1-mental-model)
2. [TL;DR](#2-tldr-the-whole-loop)
3. [Prerequisites & hardware](#3-prerequisites--hardware)
4. [Phase 0 — one-time trainer setup](#4-phase-0--one-time-trainer-setup)
5. [Phase 1 — define the roster](#5-phase-1--define-the-roster)
6. [Phase 2 — generate datasets in ComfyUI](#6-phase-2--generate-datasets-in-comfyui)
   - [6g. RECOMMENDED bootstrap — Qwen-Image-Edit (`IL_DatasetEdit`)](#6g-recommended-bootstrap--qwen-image-edit-il_datasetedit)
7. [Phase 3 — curate](#7-phase-3--curate)
8. [Phase 4 — caption + train](#8-phase-4--caption--train)
9. [Phase 5 — use the LoRA](#9-phase-5--use-the-lora)
10. [Tuning reference](#10-tuning-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [File & setting reference](#12-file--setting-reference)
13. [Why it's built this way](#13-why-its-built-this-way)

---

## 1. Mental model

- **One character = one dataset = one LoRA file.** Five characters = five independent LoRAs that
  never interfere. At render time you toggle whichever you want in the LoRA bank.
- **The chicken-and-egg** (you can't make consistent images to train on, but you need consistent
  images to train) is solved by **bootstrapping**: generate ONE good "hero" portrait, propagate
  it onto many varied poses/angles/outfits, curate, train.
- **The dataset doesn't need pixel-perfect faces** — it needs images that are *recognizably the
  same person*. Curation drops the outliers; the trained LoRA averages the rest into one stable
  identity. Don't chase a perfect clone in the dataset.

**Two ways to propagate the hero (pick per character):**

| Route | Best for | How it propagates | Section |
|---|---|---|---|
| **Qwen-Image-Edit** (`IL_DatasetEdit_<name>`) | **Fully-original characters** (recommended; default) | An image-edit model **re-poses the whole hero** (face + body + outfit) into new angles/poses while holding identity + your art style | [§6g](#6g-recommended-bootstrap--qwen-image-edit-il_datasetedit) |
| **Hero + light IPAdapter** (`IL_Dataset_<name>`) | SDXL-only / no downloads; opt-in via `hero_graph: True` | A fixed-seed hero feeds a light IPAdapter that locks the **face crop** while a wildcard prompt varies the body | [§6](#6-phase-2--generate-datasets-in-comfyui) |

Both end the same way: curate → `train_lora.ps1`. The trained LoRA — not the dataset — delivers the
final exact face either way.

```
QWEN-EDIT route (recommended for original characters):
  hero (rendered in YOUR Illustrious checkpoint)
     └─► Qwen-Image-Edit-2511 re-poses whole figure (holds identity + style)
            └─ wildcard instruction varies angle/pose/expression ─► save ─► curate ─► train ─► LoRA

HERO + IPADAPTER route (SDXL-only, no downloads):
  hero portrait (fixed seed) ─IPAdapter face lock─┐
  raw checkpoint + wildcard prompt ─► clean base ─► face detailer (hero face) ─► save ─► curate ─► train ─► LoRA
```

## 2. TL;DR (the whole loop)

**Recommended (Qwen-Image-Edit route, original characters):**
```powershell
setup.bat --with-trainer                       # once: trainer venv (multi-GB)
powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1   # once: Qwen-Edit model stack (~23 GB)
python tools/build_il_graphs.py                # emits IL_DatasetEdit_<name> per roster char + roster.json
# In ComfyUI: render an original hero in IL_1_Base -> save it as <name>_hero.png in ComfyUI/input/.
# Open IL_DatasetEdit_<name> (hero + Save prefix already wired): batch-queue the Edit-instruction
#   seed ~40x, then curate output/dataset/<name>/ to the best 25-40.
.\tools\lora_train\train_lora.ps1 -Char <name> # captions + trains  (or train_all.ps1 for the roster)
# In any IL_* workflow: LoRA bank -> toggle <name>_v1 ON, strength ~0.75, add the trigger word.
```

**SDXL-only route (hero + IPAdapter, no extra downloads):**
```powershell
setup.bat --with-trainer
# edit CHARACTERS in tools/il_graphs/config.py, then:
python tools/build_il_graphs.py                # emits IL_Dataset_<name> per character + roster.json
# In ComfyUI: open IL_Dataset_<name>, pick a hero (Hero Seed), batch-queue the Gen Seed ~15x,
#   curate output/dataset/<name>/ down to the best 25-40.
.\tools\lora_train\train_all.ps1               # captions + trains every character that has a dataset
# In any IL_* workflow: LoRA bank -> toggle <name>_v1 ON, strength ~0.75, add the trigger word.
```

## 3. Prerequisites & hardware

- **GPU:** the kit is tuned for a 16 GB NVIDIA card (RTX 5080 / Blackwell `sm_120`). SDXL LoRA
  training fits in 16 GB with the defaults (bf16, batch 2, gradient checkpointing, unet-only).
- **Blackwell note:** `sm_120` needs CUDA 12.8+ PyTorch; the default kohya torch won't run. Phase 0
  installs `torch 2.7.0 cu128` into a dedicated venv. (Your ComfyUI venv runs cu130 — also fine.)
- **uv** must be on PATH (the trainer venv uses Python 3.11; the ML stack needs ≤3.12, system Python is too new).
- **Models on disk:** the default base checkpoint `oneObsession_v19Atypical.safetensors` in
  `models/checkpoints/`, and the IPAdapter PLUS-FACE model + CLIP-ViT-H (same ones IL_IPAdapter uses).

## 4. Phase 0 — one-time trainer setup

The trainer is **kohya-ss/sd-scripts**, vendored as a submodule at `tools/sd-scripts` (a fresh
`setup.bat` initializes it in phase [2/6]). Its Python env lives at `tools/lora_train/.venv`,
separate from ComfyUI's venv. Provision it:

```powershell
setup.bat --with-trainer
```
This runs `scripts/install_trainer.ps1`, which (idempotently):
1. creates `tools/lora_train/.venv` via `uv` (Python 3.11),
2. installs **torch 2.7.0 + torchvision (cu128)** for Blackwell,
3. installs the sd-scripts requirements + **onnx/onnxruntime** (the WD14 tagger needs them; they're
   commented out in sd-scripts' own requirements) + **prodigyopt** (the optimizer),
4. runs `accelerate config default`,
5. prints the GPU torch sees.

Re-provision just the trainer (without a full setup):
`powershell -ExecutionPolicy Bypass -File scripts\install_trainer.ps1`

**Sanity check** (GPU compute + sd-scripts + tagger imports all work):
```powershell
tools\lora_train\.venv\Scripts\python.exe tools\lora_train\verify_env.py
```
Expect `GPU bf16 matmul`, `sd-scripts library import`, `onnxruntime`, `prodigyopt`, `accelerate` all `[+]`.

## 5. Phase 1 — define the roster

Edit the **`CHARACTERS`** dict in [`tools/il_graphs/config.py`](../il_graphs/config.py) — one entry
per character:

Every entry always gets a **`roster.json`** line (name/trigger/prune — the trainer's source of truth)
and a recommended **`IL_DatasetEdit_<name>`** Qwen-Image-Edit graph. The old hero+IPAdapter
`IL_Dataset_<name>` graph is emitted **only** if you set `hero_graph: True`.

```python
CHARACTERS = {
    "aria": {
        "id": "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles",  # identity ONLY
        "outfit": "cream knit sweater, blue jeans",   # OLD route only (signature clothes)
        "prune": "",            # exact tags to BAKE into the trigger ("" = leave promptable)
        "hero": "aria_hero.png" # file in ComfyUI/input/ pre-filled into IL_DatasetEdit_aria
        # "trigger": "ariachar" # optional; defaults to "<name>char"
    },
    # minimal entry: just identity + trigger/prune via defaults; hero defaults to "<name>_hero.png".
    "kael": { "id": "1boy, solo", "prune": "" },
    # OLD hero+IPAdapter route too (and the danbooru base path): set hero_graph=True.
    "nyx": { "id": "1girl, solo", "outfit": "casual hoodie, jeans", "prune": "",
             "hero_graph": True, "base": "ganyu (genshin impact)", "vary_outfit": False },
}
```

Field-by-field:
- **`id`** — identity tags ONLY: hair (colour/length/style), eyes, face marks, body. **No outfit.**
  (Mainly used by the OLD route's prompts; the edit route gets identity from your hero image.)
- **`hero`** — *(edit route)* filename in `ComfyUI/input/` of this character's hero portrait,
  pre-filled into `IL_DatasetEdit_<name>`'s LoadImage. `""` → defaults to `<name>_hero.png`.
- **`prune`** — exact tags `train_lora` strips from captions so they fold into the trigger (stronger
  identity lock). `""` keeps identity tags promptable. See [Phase 4](#8-phase-4--caption--train).
- **`hero_graph`** — *(optional, default `False`)* also emit the OLD `IL_Dataset_<name>`
  (hero+IPAdapter / `base` danbooru) graph. Leave off if you only use the edit route.
- **`outfit`** / **`vary_outfit`** — *(OLD route only)* signature vs swappable clothes for the
  hero+IPAdapter prompts. `vary_outfit: True` rolls the `__outfit__` wildcard. Ignored by the edit route.
- **`base`** — *(OLD route only)* a known **danbooru character tag** prepended to the prompt. When set, the
  tag carries a consistent face, so the generator drops the hero portrait + IPAdapter entirely
  (**pure-text, no-drift** path — Illustrious knows Danbooru-2024). When `""` (default), identity
  comes from the in-graph hero + light IPAdapter (original-face route). Paste the tag **raw**, parens
  and all (e.g. `ganyu (genshin impact)`) — the generator escapes the parens so CLIP doesn't read
  them as prompt weights. Keep `id` minimal when using `base` so the tag dominates the face. (This
  shapes *generation* only; trigger/prune are unchanged. To bake the base identity into your trigger,
  add the danbooru tag to `prune` at caption time — see [Phase 4](#8-phase-4--caption--train).)
- **`trigger`** — the caption keyword (default `<name>char`, e.g. `ariachar`). Use a *rare* string.

Then **regenerate**:
```powershell
python tools/build_il_graphs.py
```
This writes one **`IL_DatasetEdit_<name>`** workflow per entry (e.g. `IL_DatasetEdit_aria`) to
`user/default/workflows/`, plus `tools/lora_train/roster.json` (name / trigger / prune) that the
train scripts read. Entries with `hero_graph: True` also get the OLD `IL_Dataset_<name>` graph.
(Stale dataset graphs from removed/renamed characters are pruned automatically on regenerate.)

> **The training bridge is folder-based:** any dataset workflow just needs to save to
> `output/dataset/<name>/`. `train_lora.ps1 -Char <name>` and `train_all.ps1` (which iterates
> `roster.json`) then pick it up — independent of which workflow produced the images.

## 6. Phase 2 — generate datasets in ComfyUI

> **Which route?** The **recommended** path is **[§6g — Qwen-Image-Edit (`IL_DatasetEdit_<name>`)](#6g-recommended-bootstrap--qwen-image-edit-il_datasetedit)**.
> §6a–6f below document the **opt-in** hero+IPAdapter route (`IL_Dataset_<name>`, emitted only when a
> roster entry sets `hero_graph: True`). Both save to `output/dataset/<name>/` and train identically.

### 6a. Start & open (hero+IPAdapter route)
1. Launch ComfyUI: run **`run_comfy.bat`**; open `http://127.0.0.1:8188`.
2. Open the workflow **`IL_Dataset_<name>`** from the Workflows menu/sidebar (needs `hero_graph: True`).
   **If a graph is already open, re-open it after any regenerate** — ComfyUI does not auto-refresh a
   loaded graph from disk.

### 6b. Pick the hero face (sets the locked identity)
- Locate the **`Hero Seed (fixed = same face)`** node and the **`HERO preview`** node (top-left).
- To browse faces: set Hero Seed's control to **randomize**, click **Queue** a few times, watch
  **HERO preview**.
- When you like a face, set Hero Seed back to **fixed** (keep that number). That portrait is now the
  identity anchor for every image.

### 6c. Generate the set
- Confirm **Hero Seed = fixed** and **Gen Seed (reroll = variety) = randomize**.
- Set the **batch count** (the number beside Queue) to ~15 and Queue once → 15 runs × batch 4 =
  **~60 images**, saved straight to **`output/dataset/<name>/<name>_00001_.png`** …
- Each is the hero's face in a different pose / camera angle / framing / expression (and outfit, if
  `vary_outfit`).

### 6d. Graph anatomy (what each group does)
> The **Hero portrait** and **IPAdapter face lock** groups exist only when `base` is **empty**. With
> `base` set (text-only path) both are omitted and the face detailer runs on the raw checkpoint —
> the danbooru tag carries the identity.

| Group | Does |
|---|---|
| **Load + Seeds** | checkpoint, VAE, CLIP skip −2, seeds (base-empty: Hero = identity + Gen = variety; base-set: Gen only), negative. |
| **Hero portrait (identity source)** *(base empty)* | `identity + outfit + portrait suffix` → fixed-seed 832×1216 txt2img → **HERO preview**. The single face source. |
| **IPAdapter face lock (light)** *(base empty)* | `IPAdapterUnifiedLoader (PLUS FACE)` + `IPAdapterAdvanced` (weight **0.55**, **V only** — light, so wildcard expressions/poses still vary). Hero-identity model used **only** by the face detailer. |
| **Variation prompt** | `ImpactWildcardEncode`: `(base +) identity + (outfit\|__outfit__) + __framing__ __angle__ __pose__ __expression__`. Rolls new values each Gen Seed. |
| **Batched generation** | `Gen KSampler` on the **raw checkpoint** (clean render) batch 4 → decode. |
| **Face + Hand Detail** | Face detector + SAM2 → **FaceDetailer** (base empty: on the IPAdapter model, denoise 0.4, imposes the hero face; base set: on the raw model, denoise 0.3) with pose-neutral cond; Hand detailer on the raw model. |
| **Finish + Save** | `SaveImage` prefix `dataset/<name>/<name>` → `output/dataset/<name>/`. |

Base sampler config (identical to IL_1_Base): `euler_ancestral` / `normal` / 30 steps / CFG 5,
832×1216 hero / 1024×1024 batch, seed `1234567890`.

### 6e. Wildcards
A wildcard `__name__` in the prompt is replaced, **each queue**, by a random line from
`name.txt` in `custom_nodes/ComfyUI-Impact-Pack/wildcards/`. It's plain text substitution, not AI.
Edit those `.txt` files (one option per line) to change variety:

| Wildcard | File | Examples |
|---|---|---|
| `__pose__` | `pose.txt` | standing, sitting, running, arms crossed, leaning… |
| `__angle__` | `angle.txt` | front view, from side, profile, from behind, from above… |
| `__framing__` | `framing.txt` | full body, upper body, cowboy shot, close-up portrait… |
| `__expression__` | `expression.txt` | (ships with Impact-Pack) soft smile, neutral, surprised… |
| `__outfit__` | `outfit.txt` | only used when `vary_outfit: True` |

> These `.txt` live inside the Impact-Pack **submodule**, so they're not tracked by the fork — they
> exist on this machine but a fresh clone won't have them. (Known limitation.)

### 6f. Tuning dials (live in the UI — no regenerate needed)
- **Identity too weak / drifts** → `IPAdapter apply` node: raise `weight` 0.55 → 0.7 (or switch
  `embeds_scaling` to `K+V` for a harder lock — but that can flatten expression).
- **Face melty / plastic / overcooked** → lower `FaceDetailer` `denoise` → 0.3.
- **All faces too samey / stiff expression** → lower IPAdapter `weight` (0.55 → 0.4); the
  `__expression__` wildcard then comes through more. (Don't chase exact hero-match — varied is better
  training data; the LoRA delivers the final consistent face.)
- **Want a fixed outfit you forgot to set** → change `__outfit__` back to literal clothes in the
  Wildcard prompt, or set `vary_outfit` in the roster + regenerate.

When you find values you like, either leave them in the open graph or bake them into
`tools/il_graphs/graphs.py` so future regenerations keep them.

### 6g. RECOMMENDED bootstrap — Qwen-Image-Edit (`IL_DatasetEdit`)

This is the strongest 2026 way to build a dataset for a **fully-original** character (one that
doesn't resemble any known danbooru character, so a text tag can't carry the face). Instead of
locking a face *crop* (what IPAdapter does), an **image-edit model re-poses the entire hero** —
face, hair, body, and outfit — into new camera angles and poses, while preserving identity *and*
the hero's art style. You generate ONE good hero in your own checkpoint, then let the edit model
spin it into a varied set.

**Why it preserves your style.** Edit models are conditioned on the *input image* and only change
what the instruction asks. Because the hero is rendered in **your Illustrious checkpoint**, every
edited frame stays in that style — no "realism drift" from the edit model. (You can optionally add
an Illustrious img2img low-denoise re-skin pass afterward, but in practice it isn't needed.)

#### 6g.1 One-time setup

The model is **Qwen-Image-Edit-2511**, run as a quantized **GGUF** so it fits a 16 GB GPU. It needs
the **ComfyUI-GGUF** custom node (added as a submodule by `setup.bat`; if missing, run
`git submodule update --init custom_nodes/ComfyUI-GGUF` then install its `requirements.txt` into the
ComfyUI venv).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1
#   -Quant Q4_K_M   # smaller/faster, lower quality   (default Q5_K_M)
#   -SkipAnglesLora # skip the camera-angles LoRA
```
This idempotently downloads (≈23 GB total) into the right `models/` subfolders:

| File | → folder | Role |
|---|---|---|
| `qwen-image-edit-2511-Q5_K_M.gguf` (~15 GB) | `models/unet/` | the edit diffusion model (GGUF) |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` (~9 GB) | `models/text_encoders/` | the Qwen 2.5-VL text/vision encoder |
| `qwen_image_vae.safetensors` (~250 MB) | `models/vae/` | the Qwen-Image VAE |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` (~850 MB) | `models/loras/` | 4-step distillation (makes 16 GB practical) |
| `qwen-image-edit-2511-multiple-angles-lora.safetensors` (~295 MB) | `models/loras/` | drives camera-angle variety |

> **Quant guide (16 GB):** `Q5_K_M` is the sweet spot (usable quality, fits with the encoder
> offloaded to RAM). `Q4_K_M` if you OOM or want speed (noticeably weaker). `Q6_K` only with VRAM
> headroom. Re-run the installer with a different `-Quant` to swap; it skips files already present.

Then `python tools/build_il_graphs.py` emits one **`IL_DatasetEdit_<name>`** workflow per roster
character, with that character's hero filename and `dataset/<name>/<name>` save prefix pre-wired.

#### 6g.2 Graph anatomy (`IL_DatasetEdit_<name>`)

| Group | Does |
|---|---|
| **Qwen-Edit model + LoRAs** | `UnetLoaderGGUF` (Q5) → `LoraLoaderModelOnly` ×2 (Lightning 1.0, multiple-angles 0.8) → `ModelSamplingAuraFlow` (shift 3.1) → `CFGNorm` (1.0). The exact model-patch chain the official 2511 template uses. |
| **Encoders + hero** | `CLIPLoader` (qwen2.5-vl-7b, type `qwen_image`), `VAELoader` (qwen_image_vae), **`HERO >> LOAD`** (`LoadImage` — your hero), `FluxKontextImageScale` (scales the ref to the model's pixel budget). |
| **Instruction + encode** | **`Edit instruction`** (`ImpactWildcardProcessor`) → `TextEncodeQwenImageEditPlus` (positive: scaled hero + instruction) and a second one (negative: empty). Each conditioning passes a `FluxKontextMultiReferenceLatentMethod` node (kept ON — required for the repackaged GGUF build). `VAEEncode` turns the scaled hero into the init latent. |
| **Edit + decode** | `KSampler` (Lightning: **6 steps / cfg 1.0 / euler / simple / denoise 1.0**) → `VAEDecode`. |
| **Finish + Save** | `SaveImage` prefix `dataset/<name>/<name>` → `output/dataset/<name>/`. |

#### 6g.3 Step-by-step

1. **Make a hero.** Render one strong original portrait in **IL_1_Base** (front-ish, clean face,
   neutral-to-pleasant expression). Save it into **`ComfyUI/input/`** as the character's `hero`
   filename (defaults to **`<name>_hero.png`**, e.g. `aria_hero.png`). This single image is the
   entire identity source — pick a good one. (Set a different filename via the roster `hero` field.)
2. **Open `IL_DatasetEdit_<name>`** (re-open after any regenerate — ComfyUI caches loaded graphs).
   The **HERO >> LOAD** node and the `dataset/<name>/<name>` save prefix are already wired from the
   roster; just confirm the hero loaded (if the file wasn't there at build time, pick it now).
3. Confirm the **Edit instruction** node's `mode` is **populate** and its seed control is
   **randomize** (so each queue rolls a new pose/angle).
4. Set the **batch count** beside Queue to ~40 and **Queue once** → ~40 varied frames stream into
   `output/dataset/<name>/`. (One edit per queue — so use a higher batch count than the IPAdapter route.)
5. Proceed to [curate](#7-phase-3--curate) and [train](#8-phase-4--caption--train) unchanged
   (`train_lora.ps1 -Char <name>`, or `train_all.ps1` to do every roster character).

#### 6g.4 The edit instruction (driving variety)

The positive prompt is produced by **`ImpactWildcardProcessor`**, whose `wildcard_text` ships as:
```
same character, identical face and hair and outfit, keep the same art style, __angle__, __pose__, __expression__
```
- The **"same character, identical face/hair/outfit, keep the same art style"** preamble is the
  identity/style lock — keep it. It tells the edit model to change *only* the pose, not the person.
- `__angle__`, `__pose__`, `__expression__` are Impact-Pack wildcards (one random line each per
  queue, from `custom_nodes/ComfyUI-Impact-Pack/wildcards/*.txt`). Edit those `.txt` files to steer
  the kind of variety you want (e.g. add `from below`, `dutch angle` to `angle.txt`).
- You can also type a fixed instruction (set `mode` → `fixed`) to force one specific change while
  dialing in settings, e.g. `same character …, full body, three-quarter view, walking`.
- The **multiple-angles LoRA** reinforces camera-angle changes even when the prompt is mild.

#### 6g.5 Tuning dials (live in the graph)

| Symptom | Dial |
|---|---|
| Identity drifts across frames | lower **multiple-angles LoRA** strength (0.8 → 0.5); strengthen the "identical face/hair" preamble; use a cleaner hero |
| Too little variety / poses all similar | raise multiple-angles LoRA (0.8 → 1.0); add options to `angle.txt`/`pose.txt`; keep seed control = randomize |
| Soft / low-detail output | raise **KSampler steps** (6 → 8–10) — still with the Lightning LoRA; cfg can stay 1.0 |
| Style drifts from your checkpoint | ensure the hero was rendered in your checkpoint; optionally add an IL img2img re-skin (denoise 0.25–0.35) after |
| Too slow / OOM on 16 GB | re-run `install_qwen_edit.ps1 -Quant Q4_K_M`; close other GPU apps |
| Output too zoomed/cropped | the ref is auto-scaled by `FluxKontextImageScale`; add framing words (`full body`, `cowboy shot`) to the instruction |

If you settle on better defaults (LoRA strengths, steps, instruction), bake them into
`build_dataset_edit()` in `tools/il_graphs/graphs.py` so regenerations keep them.

## 7. Phase 3 — curate

Open `output/dataset/<name>/` and **delete in place** (no moving — they're already where the trainer
reads them):
- melted / distorted faces, wrong identity (off-model), bad hands cropping the face, near-duplicates.
Keep the best **25–40**, varied in pose/angle/expression. Minimum to train is **12**.

Repeat Phases 2–3 for each character (open that character's `IL_Dataset_<name>` graph).

## 8. Phase 4 — caption + train

One command per character, or the whole roster:
```powershell
.\tools\lora_train\train_lora.ps1 -Char aria      # one
.\tools\lora_train\train_all.ps1                  # every character that has a curated dataset
```

### What `train_lora.ps1` does
1. **Validates** `output/dataset/<Char>/` exists and has ≥ `-MinImages` (12).
2. **Auto-captions** (if no `.txt` present): runs the **WD14 tagger**
   (`SmilingWolf/wd-v1-4-convnextv2-tagger-v2`) to write booru tags, then `prep_captions.py`
   prepends the **trigger** and removes the **prune** tags (exact match). Trigger + prune come from
   `roster.json` unless you pass `-Trigger` / `-Prune`.
3. **Derives `num_repeats`** from the target step count: `repeats = round(Steps × Batch / (images ×
   Epochs))` — so 15 or 50 images both land near the target step count.
4. **Writes** `tools/lora_train/.cache/<Char>.toml` (the sd-scripts dataset config).
5. **Trains** via `accelerate launch sdxl_train_network.py` (from the submodule dir).

### Parameters
| Param | Default | Meaning |
|---|---|---|
| `-Char` | (required) | dataset folder name + LoRA output name |
| `-Trigger` | roster / `<Char>char` | caption keyword you'll type at inference |
| `-Prune` | roster / `""` | exact tags to bake into the trigger (e.g. `"auburn hair,green eyes"`) |
| `-Base` | oneObsession | base checkpoint to train on |
| `-Dim` / `-Alpha` | 16 / 8 | LoRA rank / alpha (try 32 / 16 for visually complex characters) |
| `-Optimizer` | `prodigy` | `prodigy` \| `adamw` \| `adafactor` (see note below — **not** adamw8bit) |
| `-DCoef` | 1.0 | Prodigy `d_coef`; lower (≈0.8) to reduce overcook on small sets |
| `-Steps` | 1500 | TARGET total steps (drives repeats) |
| `-Epochs` | 10 | epochs (saves one LoRA per epoch) |
| `-Batch` | 2 | batch size (raise if you have VRAM headroom) |
| `-TrainTextEncoder` | off | also train the text encoder (stronger, more VRAM) |
| `-SkipCaption` | off | use existing captions, don't re-tag |

**Optimizer choice (community-consensus vs our defaults).** Several Illustrious guides report Prodigy
can "burn" / not suit IL well and prefer **AdamW** or **AdaFactor** with explicit LRs (UNet ~3e-4, TE
~3e-5) — though there's a pro-Prodigy camp too, so it's genuinely contested. We keep **Prodigy as the
default** (auto-LR, no extra deps) and expose `-Optimizer adamw|adafactor` to A/B it. We deliberately
**do not** offer AdamW8bit: it needs `bitsandbytes`, which is unverified on Blackwell `sm_120` (the
exact wheel pain Prodigy avoids). There is also **no `-ClipSkip`**: `sdxl_train_network.py` ignores
clip_skip for SDXL (it warns and no-ops) — the inference-side CLIP skip −2 is unrelated to training.

### sd-scripts settings (baked into the launch)
LoRA `networks.lora` dim 16 / alpha 8 · optimizer **Prodigy** (default) lr 1.0 (`decouple`,
`weight_decay=0.01`, `d_coef=$DCoef`, `use_bias_correction`, `safeguard_warmup`) — selectable via
`-Optimizer` (adamw / adafactor at LR 3e-4) · `cosine` ·
`min_snr_gamma 5` · resolution 1024 + bucketing 768–1280 · **bf16** · batch 2 ·
`gradient_checkpointing` · `cache_latents_to_disk` · **`--sdpa`** (no xformers — avoids Blackwell
wheel pain) · `no_half_vae` · `--network_train_unet_only` (unless `-TrainTextEncoder`) · seed 42.
`PYTHONUTF8=1` is set so sd-scripts' unicode progress doesn't crash the cp1252 console.

**Output:** `models/loras/<Char>_v1.safetensors` plus one checkpoint per epoch
(`<Char>_v1-000001.safetensors` …).

### Prune: bake vs promptable
- A tag **kept** in captions → the model ties that look to the word → **promptable** (changeable).
- A tag **pruned** (and visually constant) → folds into the **trigger** → always appears, harder to change.
- **Community default for a fixed character: prune the intrinsic identity tags** (hair colour/length,
  eye colour, face marks) so they bake into the trigger and you don't have to type them. We ship
  `prune=""` (promptable) as the conservative default; for a locked signature character, populate it.
- **Using a `base` danbooru tag?** After captioning, the WD14 tagger will re-emit that character's
  tags; add the base tag (and its signature features) to `-Prune` so the borrowed identity folds into
  *your* trigger instead of staying tied to the danbooru character name.
- Signature outfit you want locked → add the outfit tags to `prune`. Swappable outfit → keep them.
- Note: prune is exact-match against WD14 output, so `"long hair"` matches the tag `long hair`, not
  `(long wavy auburn hair:1.1)`. It's optional fine-tuning — identity is learned from the consistent
  dataset regardless.

## 9. Phase 5 — use the LoRA

1. Open any IL_* workflow (IL_1_Base, IL_5_Max, …). Every graph has a **`LoRA bank`** node
   (`Power Lora Loader`) right after the checkpoint.
2. Toggle `<Char>_v1` **ON**, strength **~0.75**.
3. Put the **trigger word** (`ariachar`) in the Positive prompt.
4. Identity now flows through the base render **and** every detail pass automatically (the bank sits
   upstream of everything).

**Pick the best epoch:** in IL_1_Base, XY-plot LoRA strength {0.5, 0.75, 0.9} × a few seeds with the
trigger, across the per-epoch files. Choose the epoch/strength that holds identity without frying
the model's style.

## 10. Tuning reference

| Goal | Dial |
|---|---|
| Dataset face drifts (base-empty) | IPAdapter `weight` ↑ (0.55→0.7) — but don't over-lock; varied ≠ wrong |
| Dataset face drifts (base set) | pick a more iconic danbooru `base` tag, or trim `id` so the tag dominates |
| Dataset face melty | FaceDetailer `denoise` ↓ (0.5→0.4) |
| LoRA identity weak at inference | train strength ↑ (0.75→0.9), add `-TrainTextEncoder`, more steps |
| LoRA under-baked / low fidelity | `-Dim 32 -Alpha 16` (more capacity for complex characters) |
| LoRA burns / overcooked | `-DCoef 0.8` (Prodigy) or `-Optimizer adamw` (3e-4); fewer steps |
| LoRA overfits dataset poses | fewer `-Steps`/`-Epochs`, more varied dataset |
| LoRA fries style / too strong | LoRA bank strength ↓ (0.75→0.6), lower `-Dim`/`-Alpha` |
| Swappable outfit | `vary_outfit: True` in roster + keep outfit tags (don't prune) |
| Hard identical face (rarely worth it) | ReActor face-swap is installed (submodule) but OFF — it freezes one expression + looks uncanny, and hurts training variety. Avoid for datasets. |
| (Qwen-Edit) identity drifts across frames | lower multiple-angles LoRA (0.8→0.5); keep the "identical face/hair" preamble; use a cleaner hero — see [§6g.5](#6g5-tuning-dials-live-in-the-graph) |
| (Qwen-Edit) poses too similar | raise multiple-angles LoRA (→1.0); add lines to `angle.txt`/`pose.txt` |
| (Qwen-Edit) soft output | KSampler steps 6→8–10 (keep Lightning, cfg 1.0) |
| (Qwen-Edit) too slow / OOM (16 GB) | `install_qwen_edit.ps1 -Quant Q4_K_M`; close other GPU apps |

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Node/changes not showing in ComfyUI | A loaded graph is cached — **re-open** the workflow after regenerating. |
| Images for all characters in one folder | Old prefix bug; ensure SaveImage prefix is `dataset/<name>/<name>` (regenerate). |
| Render looks soft / washed / flat | IPAdapter was on the whole base — fixed: base is raw, IPAdapter is face-only. |
| Final face differs from hero | Weak face lock — raise IPAdapter weight / FaceDetailer denoise (see §6f). |
| `__pose__` etc. appear literally in the image | Wildcard file missing or wrong path; files go in `custom_nodes/ComfyUI-Impact-Pack/wildcards/`; reload graph. |
| `train_lora.ps1` "no dataset" | Generate to `output/dataset/<Char>/` first (SaveImage prefix `dataset/<Char>`). |
| "only N images (need ≥12)" | Generate/curate more, or lower `-MinImages`. |
| kohya errors `sm_120 not supported` | Trainer torch isn't cu128 — re-run `scripts/install_trainer.ps1`. |
| `UnicodeEncodeError` (cp1252) during train | `PYTHONUTF8=1` (the scripts set it; set it manually if you run sd-scripts directly). |
| OOM during training | drop `-Batch 1`, keep `--network_train_unet_only`. |
| `ImpactWildcardEncode` missing/red | Impact-Pack not loaded — `setup.bat` / `install_node_reqs.ps1`. |
| (Qwen-Edit) `UnetLoaderGGUF` missing/red | ComfyUI-GGUF not loaded — `git submodule update --init custom_nodes/ComfyUI-GGUF` + install its `requirements.txt`. |
| (Qwen-Edit) model not in a dropdown | not downloaded — run `scripts/install_qwen_edit.ps1`; confirm it landed in the listed `models/` subfolder. |
| (Qwen-Edit) edited frame ignores the hero | check `HERO >> LOAD` points at your image and it feeds **image1** on both encoders; keep the reference-method nodes ON. |
| (Qwen-Edit) output not anime / off-style | the hero wasn't rendered in your checkpoint — re-make it in IL_1_Base; or add an IL img2img re-skin pass. |

## 12. File & setting reference

```
tools/
  il_graphs/config.py         CHARACTERS roster, base ckpt/VAE/sampler, REF_SUFFIX
  il_graphs/graphs.py         build_dataset() — IL_Dataset_<name>; build_dataset_edit() — IL_DatasetEdit
  il_graphs/templates.py      node schemas (harvest + EXTRA_TEMPLATES incl. the Qwen-Edit nodes)
  build_il_graphs.py          regenerate all IL_* workflows + roster.json
  sd-scripts/                 kohya trainer (submodule)
  lora_train/
    README.md                 this file
    .venv/                    trainer venv (uv, py3.11, torch cu128)   [gitignored]
    roster.json               name/trigger/prune manifest             [gitignored, generated]
    .cache/<char>.toml        generated dataset configs               [gitignored]
    prep_captions.py          trigger-prepend + prune
    train_lora.ps1            caption + train one character
    train_all.ps1             train the whole roster
    verify_env.py             venv sanity check
custom_nodes/ComfyUI-Impact-Pack/wildcards/   pose/angle/framing/outfit/expression .txt
custom_nodes/ComfyUI-GGUF/     GGUF loader node (submodule; required by IL_DatasetEdit)
output/dataset/<name>/        your generated + curated images
models/loras/<name>_v1.safetensors        trained output
models/unet/qwen-image-edit-2511-Q5_K_M.gguf          Qwen-Edit diffusion model   [gitignored]
models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors  Qwen-Edit encoder     [gitignored]
models/vae/qwen_image_vae.safetensors                 Qwen-Edit VAE               [gitignored]
models/loras/Qwen-Image-Edit-2511-Lightning-*.safetensors   4-step distill LoRA   [gitignored]
models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors  angles LoRA   [gitignored]
scripts/install_trainer.ps1   builds the trainer venv (setup.bat --with-trainer)
scripts/install_qwen_edit.ps1 downloads the Qwen-Image-Edit-2511 stack (IL_DatasetEdit)
```

Key defaults — **training**: LoRA dim 16 / alpha 8 / Prodigy / ~1500 steps.
**Hero/IPAdapter route**: checkpoint `oneObsession_v19Atypical` · VAE `sdxl_vae_f16_fix` · CLIP skip −2 ·
CFG 5 · sampler `euler_ancestral`/`normal`/30 · seed `1234567890` · IPAdapter face lock 0.55 / V only ·
FaceDetailer denoise 0.4.
**Qwen-Edit route** (`IL_DatasetEdit`): GGUF Q5 · Lightning **6 steps / cfg 1.0 / euler / simple** ·
ModelSamplingAuraFlow shift 3.1 · CFGNorm 1.0 · multiple-angles LoRA 0.8 · reference method `index_timestep_zero`.

## 13. Why it's built this way

- **Clean base + face-only identity lock.** Putting IPAdapter on the whole render washes/softens it.
  So the base samples on the raw checkpoint (native quality) and the hero face is imposed only in the
  FaceDetailer crop. This is the lesson from the earlier comic work: the raw text2img base renders
  best; lock identity *after*, per face.
- **Light "V only" face-lock, not a hard swap.** A training set needs *recognizably the same person
  with varied expressions*, not identical faces. A light V-only IPAdapter (0.55) on the face keeps
  identity consistent while the `__expression__`/`__pose__` wildcards still vary. Stronger K+V froze
  expression; a ReActor face-swap froze it harder and looked uncanny — both are wrong for datasets.
  The trained LoRA, not the dataset, produces the final exact face.
- **Hero portrait as the single anchor.** One fixed-seed face, propagated, beats hoping every text
  gen lands the same face.
- **Roster + per-character graphs.** Adding a character is one config entry, not edited scripts —
  scales to N characters with no copy-paste.
- **Two venvs.** sd-scripts needs Python ≤3.12 + a pinned torch, so it can't share ComfyUI's venv.
- **Prodigy + `--sdpa`.** Prodigy auto-tunes LR (no bitsandbytes needed); `--sdpa` sidesteps
  xformers wheels on Blackwell.
- **Edit-model bootstrap > face-crop lock.** IPAdapter only conditions the *face crop*; an image-edit
  model (Qwen-Image-Edit-2511) re-poses the *whole figure* from one hero, so the dataset gets real
  pose/angle variety with the same person — the modern, higher-consistency way to bootstrap an
  original character. Rendering the hero in your own checkpoint keeps the edited frames on-style.
- **GGUF + Lightning for 16 GB.** The 20B edit model only fits a 16 GB card quantized (GGUF Q5); the
  4-step Lightning LoRA (run at 6 steps, cfg 1.0) keeps generation fast despite the text encoder
  offloading to RAM. This is why the edit route is practical on consumer hardware at all.

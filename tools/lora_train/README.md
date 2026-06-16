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
   - [Where to edit what (quick map)](#where-to-edit-what-quick-map)
6. [Phase 2 — generate datasets in ComfyUI (Qwen-Image-Edit)](#6-phase-2--generate-datasets-in-comfyui-qwen-image-edit)
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

**The dataset route: Qwen-Image-Edit (`IL_DatasetEdit_<name>`).** The generator emits one
self-contained, two-stage graph per roster character. An image-edit model **re-poses the whole
hero** (face + body + outfit) into new angles/poses/scenes while holding identity + your art style.
It works for fully-original characters (no danbooru anchor needed). Curate → `train_lora.ps1`. The
trained LoRA — not the dataset — delivers the final exact face.

```
QWEN-EDIT route (self-contained, two stages):
  STAGE 1: text2img the hero from the character's id tags in YOUR checkpoint (reroll Hero Seed, pick a face)
     └─► STAGE 2: Qwen-Image-Edit-2511 re-poses that whole figure (holds identity + style)
            └─ wildcard instruction varies framing/angle/pose/expression/background/lighting
               ─► save ─► curate ─► train ─► LoRA
```

## 2. TL;DR (the whole loop)

```powershell
setup.bat --with-trainer                       # once: trainer venv (multi-GB)
powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1   # once: Qwen-Edit model stack (~23 GB)
python tools/build_il_graphs.py                # emits IL_DatasetEdit_<name> per roster char + roster.json
# In ComfyUI open IL_DatasetEdit_<name> (self-contained):
#   STAGE 1: reroll Hero Seed -> pick the face in HERO preview (rendered from the id, your style).
#   STAGE 2: batch-queue the Edit-instruction seed ~40x -> output/dataset/<name>/; curate to 25-40.
.\tools\lora_train\train_lora.ps1 -Char <name> # captions + trains  (or train_all.ps1 for the roster)
# In any IL_* workflow: LoRA bank -> toggle <name>_v1 ON, strength ~0.75, add the trigger word.
```

## 3. Prerequisites & hardware

- **GPU:** the kit is tuned for a 16 GB NVIDIA card (RTX 5080 / Blackwell `sm_120`). SDXL LoRA
  training fits in 16 GB with the defaults (bf16, batch 2, gradient checkpointing, unet-only).
- **Blackwell note:** `sm_120` needs CUDA 12.8+ PyTorch; the default kohya torch won't run. Phase 0
  installs `torch 2.7.0 cu128` into a dedicated venv. (Your ComfyUI venv runs cu130 — also fine.)
- **uv** must be on PATH (the trainer venv uses Python 3.11; the ML stack needs ≤3.12, system Python is too new).
- **Models on disk:** the default base checkpoint `oneObsession_v19Atypical.safetensors` in
  `models/checkpoints/` (renders the Stage-1 hero), plus the Qwen-Image-Edit stack that
  `scripts/install_qwen_edit.ps1` downloads (see [§6.1](#61-one-time-setup)).

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
per character. Every entry gets a **`roster.json`** line (name/trigger/prune — the trainer's source
of truth) and an **`IL_DatasetEdit_<name>`** Qwen-Image-Edit graph.

```python
CHARACTERS = {
    "aria": {
        "id": "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles",  # identity ONLY
        "prune": "",            # exact tags to BAKE into the trigger ("" = leave promptable)
        # "trigger": "ariachar" # optional; defaults to "<name>char"
    },
    # minimal entry: just identity; trigger defaults to kaelchar.
    "kael": { "id": "1boy, solo, (tousled black hair:1.1), (sharp blue eyes:1.1)", "prune": "" },
    # optional signature outfit (worn by the Stage-1 hero):
    "nyx": { "id": "1girl, solo, (silver bob hair:1.1), (violet eyes:1.1)",
             "outfit": "casual hoodie, jeans", "prune": "" },
}
```

Field-by-field:
- **`id`** — identity tags ONLY: hair (colour/length/style), eyes, face marks, body. **No outfit.**
  This is the **Stage-1 hero prompt** in `IL_DatasetEdit_<name>` — be specific and weight the
  face-defining tags (`(green eyes:1.1)`). The hero is rendered from it, then re-posed.
- **`outfit`** — *(optional)* signature clothes, appended to the Stage-1 hero prompt so the hero
  wears them (the edit then keeps that outfit across poses). Leave `""` to let the checkpoint pick.
- **`prune`** — exact tags `train_lora` strips from captions so they fold into the trigger (stronger
  identity lock). `""` keeps identity tags promptable. See [Phase 4](#8-phase-4--caption--train).
- **`trigger`** — the caption keyword (default `<name>char`, e.g. `ariachar`). Use a *rare* string.

Then **regenerate**:
```powershell
python tools/build_il_graphs.py
```
This writes one **`IL_DatasetEdit_<name>`** workflow per entry (e.g. `IL_DatasetEdit_aria`) to
`user/default/workflows/`, plus `tools/lora_train/roster.json` (name / trigger / prune) that the
train scripts read. (Stale dataset graphs from removed/renamed characters — and any legacy
`IL_Dataset_<name>` from the retired hero+IPAdapter route — are pruned automatically on regenerate.)

> **The training bridge is folder-based:** any dataset workflow just needs to save to
> `output/dataset/<name>/`. `train_lora.ps1 -Char <name>` and `train_all.ps1` (which iterates
> `roster.json`) then pick it up — independent of which workflow produced the images.

### Where to edit what (quick map)

Two kinds of change: **config/graph** edits need a `python tools/build_il_graphs.py` regenerate (then
re-open the workflow); **wildcard `.txt`** edits are *live* — just re-open/queue the graph, no regenerate.

| Want to change | Edit | After |
|---|---|---|
| **Add / remove a character** | `CHARACTERS` in [`tools/il_graphs/config.py`](../il_graphs/config.py) | regenerate |
| **Character identity** (face/hair/eyes) | that entry's `id` (Stage-1 hero prompt) | regenerate |
| **Trigger / pruned tags** | `trigger` / `prune` in the entry | regenerate (rewrites `roster.json`) |
| **Poses** | `custom_nodes/ComfyUI-Impact-Pack/wildcards/pose.txt` | reload graph |
| **Camera angles** | `…/wildcards/angle.txt` | reload graph |
| **Expressions** | `…/wildcards/expression.txt` | reload graph |
| **Framing** (full body / close-up) | `…/wildcards/framing.txt` | reload graph |
| **Backgrounds / scenes** | `…/wildcards/background.txt` | reload graph |
| **Lighting** | `…/wildcards/lighting.txt` | reload graph |
| **The edit instruction template** | `wtext` in `build_dataset_edit()` ([`graphs.py`](../il_graphs/graphs.py)) | regenerate |
| **Hero render** (checkpoint / sampler / steps / size) | `CKPT`, `BASE_*`, `REF_SUFFIX` in `config.py` | regenerate |
| **Qwen-Edit knobs** (LoRA strengths, KSampler steps) | `build_dataset_edit()` in `graphs.py` | regenerate |
| **Qwen-Edit quant** (VRAM/speed) | `scripts/install_qwen_edit.ps1 -Quant Q4_K_M` | re-run installer |
| **Training params** (rank / optimizer / steps) | `train_lora.ps1` flags — `-Dim`, `-Optimizer`, `-Steps`, … | n/a |

> ⚠️ The wildcard `.txt` files live inside the **Impact-Pack submodule** (untracked by this fork), so
> they exist on this machine but a fresh clone won't have them. Editing them is per-machine.
> One wildcard line = one random option per queue; add lines (e.g. `from below`, `dutch angle`) to widen variety.

## 6. Phase 2 — generate datasets in ComfyUI (Qwen-Image-Edit)

`IL_DatasetEdit_<name>` is the strongest 2026 way to build a dataset for a **fully-original**
character (one that doesn't resemble any known danbooru character, so a text tag can't carry the
face). The graph is **self-contained and two-stage**: **Stage 1** renders ONE hero from the
character's `id` tags in your own checkpoint (you reroll a fixed **Hero Seed** and pick the face you
like); **Stage 2** lets an **image-edit model re-pose that entire hero** — face, hair, body, outfit
— into new camera angles, poses, and scenes, preserving identity *and* the art style. So a brand-new
character needs **no pre-existing image**: the graph makes the hero, then propagates it.

**Bootstrap, explained.** "Text drifts" is about getting the *same* face across *many* gens — but
you only need **one** face, and a single text2img gives you one. Stage 1 is that single render
(reroll the Hero Seed until you like it); Stage 2 does the consistency work by editing that one
image, not by re-prompting.

**Why it preserves your style.** Edit models are conditioned on the *input image* and only change
what the instruction asks. Because the hero is rendered in **your Illustrious checkpoint** (Stage 1),
every edited frame stays in that style — no "realism drift" from the edit model. (You can optionally
add an Illustrious img2img low-denoise re-skin pass afterward, but in practice it isn't needed.)

### 6.1 One-time setup

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
character — Stage-1 prompt (the `id`), Stage-2 model, and `dataset/<name>/<name>` save prefix all
pre-wired.

### 6.2 Graph anatomy (`IL_DatasetEdit_<name>`)

| Group | Does |
|---|---|
| **STAGE 1 — Hero generator (Illustrious)** | `CheckpointLoaderSimple` + `CLIPSetLastLayer −2` + the character's `id` prompt + `KSampler` (euler_a/normal/30/cfg5, fixed **Hero Seed**) → `VAEDecode` → **HERO preview**. Renders the single hero in your style — no input image needed. |
| **STAGE 2 — Qwen-Edit model + LoRAs** | `UnetLoaderGGUF` (Q5) → `LoraLoaderModelOnly` ×2 (Lightning 1.0, multiple-angles 1.0) → `ModelSamplingAuraFlow` (shift 3.1) → `CFGNorm` (1.0). The exact model-patch chain the official 2511 template uses. |
| **Encoders + scale** | `CLIPLoader` (qwen2.5-vl-7b, type `qwen_image`), `VAELoader` (qwen_image_vae), `FluxKontextImageScale` (scales the hero to the model's pixel budget — fed by Stage 1). |
| **Instruction + encode** | **`Edit instruction`** (`ImpactWildcardProcessor`, **mode `fixed`** — see [§6.4](#64-the-edit-instruction-driving-variety)) → `TextEncodeQwenImageEditPlus` (positive: scaled hero + instruction) and a second one (negative: empty). Each conditioning passes a `FluxKontextMultiReferenceLatentMethod` node (kept ON — required for the repackaged GGUF build). `VAEEncode` turns the scaled hero into the init latent. |
| **Edit + decode** | `KSampler` (Lightning: **6 steps / cfg 1.0 / euler / simple / denoise 1.0**) → `VAEDecode`. |
| **Finish + Save** | `SaveImage` prefix `dataset/<name>/<name>` → `output/dataset/<name>/`. |

### 6.3 Step-by-step

1. **Open `IL_DatasetEdit_<name>`** (re-open after any regenerate — ComfyUI caches loaded graphs).
   Everything is pre-wired from the roster.
2. **Stage 1 — pick the face.** Reroll the **Hero Seed** and watch **HERO preview** until you like
   the rendered face (it comes from this character's `id` tags in your checkpoint). Then leave Hero
   Seed **fixed** on that value — that single image is now the identity anchor.
3. **Stage 2 — confirm variety.** The **Edit instruction** node is `mode: fixed` with its seed
   control **randomize**. Each queue advances the seed, and the wildcards expand in the node backend
   on that seed — so every frame rolls a new framing/angle/pose/expression/background/lighting.
4. Set the **batch count** beside Queue to ~40 and **Queue once** → ~40 varied frames stream into
   `output/dataset/<name>/`. (One edit per queue — so use a higher batch count than a batched txt2img.)
5. Proceed to [curate](#7-phase-3--curate) and [train](#8-phase-4--caption--train)
   (`train_lora.ps1 -Char <name>`, or `train_all.ps1` to do every roster character).

### 6.4 The edit instruction (driving variety)

The positive prompt is produced by **`ImpactWildcardProcessor`**, whose text ships as:
```
Change the shot to __framing__ from __angle__. Re-pose the character to __pose__, __expression__.
Set the scene: __background__, __lighting__. Keep the exact same character (identical face,
hairstyle and outfit) and the same anime art style.
```
- **Lead with the change.** Qwen-Image-Edit is conservative: it changes only what the instruction
  asks. Leading with the imperative edit verbs ("Change the shot … Re-pose …") makes the camera/pose
  actually move; the identity + style lock is a concise trailing clause so it doesn't drown the edit.
- **`mode: fixed` is deliberate.** The `ImpactWildcardProcessor` backend only expands wildcards found
  in `populated_text` (`doit()` → `process(populated_text, seed)`); the "populate" copy of
  `wildcard_text` → `populated_text` is a browser-JS step that **does not run on a headless API POST**
  and can fire only once per Queue in the UI. So the generator puts the same wildcard string in *both*
  text boxes and sets mode `fixed`: the backend then re-rolls a fresh instruction **every execution**,
  keyed on the seed (which `control_after_generate=randomize` advances per batch item / per POST). This
  is what fixes the old "every frame is a side-view sitting pose" sameness.
- `__framing__/__angle__/__pose__/__expression__/__background__/__lighting__` are Impact-Pack
  wildcards (one random line each per roll, from `custom_nodes/ComfyUI-Impact-Pack/wildcards/*.txt`).
  Edit those `.txt` files to steer the kind of variety you want (add `from below`, `dutch angle`, etc.).
- The **multiple-angles LoRA** (strength 1.0) reinforces camera-angle changes even when the prompt is mild.

### 6.5 Tuning dials (live in the graph)

| Symptom | Dial |
|---|---|
| Identity drifts across frames | lower **multiple-angles LoRA** strength (1.0 → 0.6); trim the scene clause (`__background__/__lighting__`); use a cleaner hero |
| Too little variety / poses all similar | confirm the Edit-instruction seed control = randomize; add options to `angle.txt`/`pose.txt`/`framing.txt`; keep the multiple-angles LoRA at 1.0 |
| Soft / low-detail output | raise **KSampler steps** (6 → 8–10) — still with the Lightning LoRA; cfg can stay 1.0 |
| Style drifts from your checkpoint | ensure the hero was rendered in your checkpoint; optionally add an IL img2img re-skin (denoise 0.25–0.35) after |
| Too slow / OOM on 16 GB | re-run `install_qwen_edit.ps1 -Quant Q4_K_M`; close other GPU apps |
| Output too zoomed/cropped | the ref is auto-scaled by `FluxKontextImageScale`; the instruction already leads with `__framing__` — add framing words to `framing.txt` |

If you settle on better defaults (LoRA strengths, steps, instruction), bake them into
`build_dataset_edit()` in `tools/il_graphs/graphs.py` so regenerations keep them.

## 7. Phase 3 — curate

Open `output/dataset/<name>/` and **delete in place** (no moving — they're already where the trainer
reads them):
- melted / distorted faces, wrong identity (off-model), bad hands cropping the face, near-duplicates.
Keep the best **25–40**, varied in pose/angle/expression/scene. Minimum to train is **12**.

Repeat Phases 2–3 for each character (open that character's `IL_DatasetEdit_<name>` graph).

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
| `-Epochs` | 4 | saves one LoRA per epoch -> N checkpoints to pick from. Total steps stay ~= `-Steps` (repeats compensate), so this is checkpoint granularity, not training amount. |
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
| Identity drifts across edited frames | lower multiple-angles LoRA (1.0→0.6); trim the instruction's scene clause; use a cleaner hero — see [§6.5](#65-tuning-dials-live-in-the-graph) |
| Poses too similar | confirm Edit-instruction seed control = randomize; add lines to `angle.txt`/`pose.txt`/`framing.txt` |
| Soft / low-detail edits | KSampler steps 6→8–10 (keep Lightning, cfg 1.0) |
| Too slow / OOM (16 GB) | `install_qwen_edit.ps1 -Quant Q4_K_M`; close other GPU apps |
| LoRA identity weak at inference | train strength ↑ (0.75→0.9), add `-TrainTextEncoder`, more steps |
| LoRA under-baked / low fidelity | `-Dim 32 -Alpha 16` (more capacity for complex characters) |
| LoRA burns / overcooked | `-DCoef 0.8` (Prodigy) or `-Optimizer adamw` (3e-4); fewer steps |
| LoRA overfits dataset poses | fewer `-Steps`/`-Epochs`, more varied dataset |
| LoRA fries style / too strong | LoRA bank strength ↓ (0.75→0.6), lower `-Dim`/`-Alpha` |

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Node/changes not showing in ComfyUI | A loaded graph is cached — **re-open** the workflow after regenerating. |
| Images for all characters in one folder | Old prefix bug; ensure SaveImage prefix is `dataset/<name>/<name>` (regenerate). |
| Every frame is the same pose/angle | Edit-instruction node must be `mode: fixed` with seed control = randomize (so the backend re-rolls per run). Headless API POSTs need the seed-randomizing runner (`convert_and_run.py`) — a raw POST keeps the saved seed. |
| `__pose__` etc. appear literally in the image | Wildcard file missing or wrong path; files go in `custom_nodes/ComfyUI-Impact-Pack/wildcards/`; reload graph. |
| `train_lora.ps1` "no dataset" | Generate to `output/dataset/<Char>/` first (SaveImage prefix `dataset/<Char>`). |
| "only N images (need ≥12)" | Generate/curate more, or lower `-MinImages`. |
| kohya errors `sm_120 not supported` | Trainer torch isn't cu128 — re-run `scripts/install_trainer.ps1`. |
| `UnicodeEncodeError` (cp1252) during train | `PYTHONUTF8=1` (the scripts set it; set it manually if you run sd-scripts directly). |
| OOM during training | drop `-Batch 1`, keep `--network_train_unet_only`. |
| `ImpactWildcardProcessor` missing/red | Impact-Pack not loaded — `setup.bat` / `install_node_reqs.ps1`. |
| `UnetLoaderGGUF` missing/red | ComfyUI-GGUF not loaded — `git submodule update --init custom_nodes/ComfyUI-GGUF` + install its `requirements.txt`. |
| Model not in a dropdown | not downloaded — run `scripts/install_qwen_edit.ps1`; confirm it landed in the listed `models/` subfolder. |
| Edited frame ignores the hero | confirm Stage-1 `Hero decode` feeds **Scale ref**, and that feeds **image1** on both encoders; keep the reference-method nodes ON. |
| Output not anime / off-style | Stage 1 renders in your checkpoint (`CKPT`) so it should match; if off, tighten `id` or add an IL img2img re-skin pass. |
| Stage-1 hero looks wrong | tighten the character's `id` tags (weight face-defining ones); reroll the Hero Seed for a better face. |

## 12. File & setting reference

```
tools/
  il_graphs/config.py         CHARACTERS roster, base ckpt/VAE/sampler, REF_SUFFIX
  il_graphs/graphs.py         build_dataset_edit() — IL_DatasetEdit_<name> (Qwen-Image-Edit)
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
custom_nodes/ComfyUI-Impact-Pack/wildcards/   framing/angle/pose/expression/background/lighting .txt
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
**Stage-1 hero**: checkpoint `oneObsession_v19Atypical` · VAE `sdxl_vae_f16_fix` · CLIP skip −2 ·
CFG 5 · sampler `euler_ancestral`/`normal`/30 · 832×1216 · seed `1234567890`.
**Qwen-Edit (Stage 2)**: GGUF Q5 · Lightning **6 steps / cfg 1.0 / euler / simple** ·
ModelSamplingAuraFlow shift 3.1 · CFGNorm 1.0 · multiple-angles LoRA 1.0 · reference method `index_timestep_zero`.

## 13. Why it's built this way

- **Edit-model bootstrap.** An image-edit model (Qwen-Image-Edit-2511) re-poses the *whole figure*
  from one hero, so the dataset gets real pose/angle/scene variety with the same person — the modern,
  higher-consistency way to bootstrap an original character. Rendering the hero in your own checkpoint
  keeps the edited frames on-style.
- **Hero as the single anchor.** One fixed-seed face, propagated, beats hoping every text gen lands
  the same face. The dataset only needs *recognizably the same person*; the trained LoRA produces the
  final exact face.
- **Backend wildcard expansion, not the browser populate step.** The instruction node runs in `fixed`
  mode with the wildcards in `populated_text`, so a fresh roll happens in the node backend every
  execution (keyed on the seed) — variety survives headless API runs and per-batch queuing, instead of
  depending on a browser-only populate pass. (This was the root cause of the early "all frames look the
  same" symptom.)
- **GGUF + Lightning for 16 GB.** The 20B edit model only fits a 16 GB card quantized (GGUF Q5); the
  4-step Lightning LoRA (run at 6 steps, cfg 1.0) keeps generation fast despite the text encoder
  offloading to RAM. This is why the edit route is practical on consumer hardware at all.
- **Roster + per-character graphs.** Adding a character is one config entry, not edited scripts —
  scales to N characters with no copy-paste.
- **Two venvs.** sd-scripts needs Python ≤3.12 + a pinned torch, so it can't share ComfyUI's venv.
- **Prodigy + `--sdpa`.** Prodigy auto-tunes LR (no bitsandbytes needed); `--sdpa` sidesteps
  xformers wheels on Blackwell.

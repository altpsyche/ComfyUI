# Character LoRA training

Train one LoRA per character so it renders **identically every time** in any IL_* workflow. Text
prompts and IPAdapter both drift; a trained LoRA is the only thing that locks an exact identity. This
kit takes you from "no images" to "trained LoRA in the LoRA bank", fully scripted — no per-character
file copies, no manual moving.

## Where to start

| You want to… | Read |
|---|---|
| Run the whole loop, commands only | **[QUICKSTART.md](QUICKSTART.md)** |
| Add a character without missing a step | **[ADD_CHARACTER.md](ADD_CHARACTER.md)** (the authoritative checklist) |
| Choose how to control clothes (outfit / keep / outfits / like) | **[CLOTHING_MODEL.md](CLOTHING_MODEL.md)** |
| Understand how the dataset images are made | **[DATASET.md](DATASET.md)** (the Qwen-Image-Edit engine) |
| Look up a knob / flag / command | **[REFERENCE.md](REFERENCE.md)** |
| Steer pose/angle/scene variety | **[WILDCARDS.md](WILDCARDS.md)** |
| Change the pipeline (don't re-discover dead ends) | **[GOTCHAS.md](GOTCHAS.md)** |
| Extend the workflow generator (code) | **[../il_graphs/ARCHITECTURE.md](../il_graphs/ARCHITECTURE.md)** |

This page owns the **concepts, setup, and the file map**. Each topic above has one owner — they
cross-link, they don't repeat.

## Mental model

- **One character = one dataset = one LoRA file.** Five characters = five independent LoRAs that never
  interfere; at render time you toggle whichever you want in the LoRA bank. (A **modular** character is
  still one LoRA, but carries several swappable outfits — see [CLOTHING_MODEL.md](CLOTHING_MODEL.md).)
- **The chicken-and-egg** (you can't make consistent images without a consistent character, but you
  need consistent images to train one) is solved by **bootstrapping**: generate ONE good "hero"
  portrait, propagate it onto many varied poses/angles/outfits, curate, train.
- **The dataset doesn't need pixel-perfect faces** — just images that are *recognizably the same
  person*. Curation drops outliers; the trained LoRA averages the rest into one stable identity. Don't
  chase a perfect clone in the dataset.
- **Identity comes from the dataset + trigger + training, not from pruning.** The `outfit` string is
  auto-baked into the trigger so the costume is always-on; identity tags stay promptable by default.

## The loop

```
characters.toml  ──build──▶  IL_DatasetEdit_<name>  ──generate──▶  output/dataset/<name>/
   (id · outfit)              (Qwen-Image-Edit)         + curate
                                                            │
                          train_lora.ps1 ◀─ roster.json ◀──┘   ──▶  models/loras/<name>_v1.safetensors
                          (caption + train)                          (toggle in any IL_* LoRA bank)
```

1. **Define** — add a `[table]` to [`../il_graphs/characters.toml`](../il_graphs/characters.toml). → [ADD_CHARACTER.md](ADD_CHARACTER.md)
2. **Generate** — `python tools/build_il_graphs.py`, open `IL_DatasetEdit_<name>`, queue ~40. → [DATASET.md](DATASET.md)
3. **Curate** — keep the on-model 25–40 (min 12), delete in place.
4. **Train** — `train_lora.ps1 -Char <name>` (or `train_all.ps1`). → [REFERENCE.md](REFERENCE.md)
5. **Use** — toggle `<name>_v1` ON in any LoRA bank, strength ~0.75, add the trigger word.

## Prerequisites & hardware

- **GPU:** tuned for a 16 GB NVIDIA card (RTX 5080 / Blackwell `sm_120`). SDXL LoRA training fits in
  16 GB with the defaults (bf16, batch 2, gradient checkpointing, unet-only).
- **Blackwell:** `sm_120` needs CUDA 12.8+ PyTorch; the default kohya torch won't run, so Phase 0
  installs `torch 2.7.0 cu128` into a dedicated venv. (Your ComfyUI venv runs cu130 — also fine.)
- **uv** on PATH (the trainer venv is Python 3.11; the ML stack needs ≤3.12, system Python is too new).
- **Models on disk:** the default base checkpoint `oneObsession_v19Atypical.safetensors` in
  `models/checkpoints/` (renders the Stage-1 hero) + the Qwen-Image-Edit stack
  ([DATASET.md](DATASET.md) installs it).

## Setup (once per machine)

```powershell
setup.bat --with-trainer                                                 # trainer venv (multi-GB)
powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1   # Qwen-Edit stack (~23 GB)
```

`setup.bat --with-trainer` runs `scripts/install_trainer.ps1`, which idempotently: creates
`tools/lora_train/.venv` via `uv` (Python 3.11) · installs **torch 2.7.0 + torchvision (cu128)** ·
the sd-scripts requirements + **onnx/onnxruntime** (the WD14 tagger needs them) + **prodigyopt** ·
runs `accelerate config default`. Re-provision alone with
`powershell -ExecutionPolicy Bypass -File scripts\install_trainer.ps1`.

The trainer is **kohya-ss/sd-scripts**, vendored as a submodule at `tools/sd-scripts`; its venv lives
at `tools/lora_train/.venv`, separate from ComfyUI's. Sanity-check it:
```powershell
tools\lora_train\.venv\Scripts\python.exe tools\lora_train\verify_env.py
```
Expect `[+]` on: GPU bf16 matmul · sd-scripts import · onnxruntime · prodigyopt · accelerate · dataset
TOML parses · wildcards present.

## Where to edit what

Two kinds of change: **config/graph** edits need a `python tools/build_il_graphs.py` regenerate (then
re-open the workflow); **wildcard `.txt`** edits are *live* — just re-open/queue the graph.

| Want to change | Edit | After |
|---|---|---|
| **Add / remove a character** | a `[table]` in [`characters.toml`](../il_graphs/characters.toml) | regenerate |
| **Same character, new locked outfit** | a `[table]` with `like = "<char>"` + its own `outfit` | regenerate (own LoRA, same identity; pin `hero_seed`) |
| **Same character, swappable outfits** (one LoRA) | a `[<char>.outfits]` table (modular) — see [CLOTHING_MODEL.md](CLOTHING_MODEL.md) | regenerate (one LoRA, `mirachar, mira_<outfit>`) |
| **Signature outfit** (auto-locked) | that table's `outfit` (auto-baked into the trigger at train) | regenerate |
| **Character identity** (face/hair/eyes) | that table's `id` (Stage-1 hero prompt) | regenerate |
| **Trigger / pruned tags** | `trigger` / `prune` in the table | regenerate (rewrites `roster.json`) |
| **Pose / angle / expression / framing / background / lighting** | the matching `custom_nodes/ComfyUI-Impact-Pack/wildcards/*.txt` | reload graph ([WILDCARDS.md](WILDCARDS.md)) |
| **The edit instruction template** | `wtext` in `build_dataset_edit()` ([`graphs.py`](../il_graphs/graphs.py)) | regenerate |
| **Hero render** (checkpoint / sampler / steps / size) | `CKPT`, `BASE_*`, `REF_SUFFIX` in [`config.py`](../il_graphs/config.py) | regenerate |
| **Qwen-Edit knobs** (LoRA strengths, KSampler steps) | `build_dataset_edit()` in [`graphs.py`](../il_graphs/graphs.py) | regenerate |
| **Qwen-Edit quant** (VRAM/speed) | `scripts/install_qwen_edit.ps1 -Quant Q4_K_M` | re-run installer |
| **Training params** (rank / optimizer / steps / LR / resolution …) | `train.toml` or `train_lora.ps1` flags — matrix in [REFERENCE.md](REFERENCE.md) | n/a |

> **The training bridge is folder-based:** any dataset workflow just needs to save to
> `output/dataset/<name>/`; `train_lora.ps1 -Char <name>` and `train_all.ps1` pick it up regardless of
> which workflow produced the images. The wildcard `.txt` files live inside the **Impact-Pack
> submodule** (untracked by this fork) — they exist on this machine but a fresh clone won't have them.

## Why it's built this way

- **Edit-model bootstrap.** An image-edit model (Qwen-Image-Edit-2511) re-poses the *whole figure* from
  one hero, giving real pose/angle/scene variety with the same person — the higher-consistency way to
  bootstrap an original character. Rendering the hero in your own checkpoint keeps frames on-style.
- **Hero as the single anchor.** One fixed-seed face, propagated, beats hoping every text gen lands the
  same face. The dataset only needs *recognizably the same person*; the LoRA produces the exact face.
- **Roster + per-character graphs.** Adding a character is one config entry, not edited scripts —
  scales to N characters with no copy-paste.
- **Data-driven training params.** Hyperparameters live in `train.toml` (layered defaults / profiles /
  per-char), not hardcoded in the `.ps1`. → [REFERENCE.md](REFERENCE.md)
- The SDXL/Qwen/Blackwell-specific choices (GGUF+Lightning for 16 GB, Prodigy + `--sdpa`, two venvs,
  CLIP-skip −2) and what NOT to undo are in [GOTCHAS.md](GOTCHAS.md).

## File & setting reference

```
tools/
  il_graphs/characters.toml   the roster (one [table] per character) — edit here to add a character
  il_graphs/config.py         loads characters.toml; base ckpt/VAE/sampler, REF_SUFFIX
  il_graphs/graphs.py         build_dataset_edit() — IL_DatasetEdit_<name> (Qwen-Image-Edit)
  il_graphs/ARCHITECTURE.md   how the generator package is built (extend it here)
  build_il_graphs.py          regenerate all IL_* workflows + roster.json (+ validates them)
  validate_workflow.py        pre-flight a workflow (models/wildcards/embeddings + rules)
  sd-scripts/                 kohya trainer (submodule)
  tests/                      pytest suite (build/golden/cfg/prep/train_config/docs)
  lora_train/
    README.md                 this file — concepts, setup, file map
    QUICKSTART.md             the whole loop, commands only
    ADD_CHARACTER.md          add-a-character checklist (authoritative procedure)
    CLOTHING_MODEL.md         outfit / keep / outfits (modular) / like — which to use
    DATASET.md                the Qwen-Image-Edit dataset engine
    REFERENCE.md              every command + the full training-param matrix
    GOTCHAS.md                dead ends + traps — read before changing the pipeline
    WILDCARDS.md              steering dataset variety
    train.toml                training hyperparams: [defaults] / [profiles.*] / [train.<char>]
    train_config.py           resolves train.toml (defaults < per-char < profile < CLI) -> JSON
    dataset_plan.py           modular: balanced num_repeats + multi-subset dataset .toml (keep_tokens=2)
    train_lora.ps1            caption + train one character (-Profile / -DryRun / param flags)
    train_all.ps1             train the whole roster
    prep_captions.py          trigger-prepend + auto-bake outfit (tolerant) + extra prune (--dry-run)
    cull_dataset.py           flag blurry / near-duplicate frames before curating (--apply)
    gen_dataset.py            headless: queue IL_DatasetEdit N times with fresh seeds (needs Export-API)
    verify_env.py             venv sanity check
    .venv/                    trainer venv (uv, py3.11, torch cu128)   [gitignored]
    roster.json               name/trigger/id/outfit/prune manifest   [gitignored, generated]
    .cache/<char>.toml        generated dataset configs               [gitignored]
custom_nodes/ComfyUI-Impact-Pack/wildcards/   framing/angle/pose/expression/background/lighting .txt
custom_nodes/ComfyUI-GGUF/     GGUF loader node (submodule; required by IL_DatasetEdit)
output/dataset/<name>/        your generated + curated images
models/loras/<name>_v1.safetensors        trained output (+ one checkpoint per epoch)
scripts/install_trainer.ps1   builds the trainer venv (setup.bat --with-trainer)
scripts/install_qwen_edit.ps1 downloads the Qwen-Image-Edit-2511 stack (see DATASET.md)
```

Key defaults — **training**: LoRA dim 16 / alpha 8 / Prodigy / ~1500 steps (full matrix:
[REFERENCE.md](REFERENCE.md)). **Stage-1 hero**: `oneObsession_v19Atypical` · VAE `sdxl_vae_f16_fix` ·
CLIP skip −2 · CFG 5 · `euler_ancestral`/`normal`/30 · 832×1216. **Qwen-Edit**: GGUF Q5 · Lightning
6 steps / cfg 1.0 (see [DATASET.md](DATASET.md)).

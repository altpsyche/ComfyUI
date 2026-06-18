# Reference — what the LoRA toolchain offers

The single "what can I tune, and where" page. Adding a character start-to-finish:
[ADD_CHARACTER.md](ADD_CHARACTER.md). Step-by-step quick loop: [QUICKSTART.md](QUICKSTART.md); the full
narrative in [README.md](README.md); traps in [GOTCHAS.md](GOTCHAS.md).

## The two configs (and how they meet)

```
  characters.toml ──build──▶ roster.json ──▶ train_lora.ps1 ─┐
   (identity / dataset)        (trigger/id/        (per char) │
   id · outfit · like ·         outfit/prune)                 ├──▶ accelerate → LoRA
   hero_seed · trigger                                        │
                                                              │
  train.toml ─────────────resolve (train_config.py)───────────┘
   (trainer hyperparams)   defaults < per-char < profile < CLI
```

- **`il_graphs/characters.toml`** — *who* to train: identity tags, signature outfit, costume variants.
  Edited to add/remove a character; `python tools/build_il_graphs.py` turns it into `roster.json` + one
  `IL_DatasetEdit_<name>` graph each.
- **`lora_train/train.toml`** — *how* to train: every hyperparameter, with named profiles and optional
  per-character overrides. Resolved by `train_config.py`. These two files are kept separate so the
  ComfyUI-side generator config and the trainer config don't couple.

## Commands

| Command | Purpose | Key flags |
|---|---|---|
| `python tools/build_il_graphs.py` | Build all IL_* workflows from `characters.toml`; writes `roster.json`; **validates every emitted graph**. | `--no-validate` (skip the guardrail) |
| `python tools/validate_workflow.py <graph>.json` | Pre-flight a workflow (models/wildcards/embeddings exist; rules: CLIP skip −2, CFG floor). | `--strict`, `--rules <path>` |
| `tools\lora_train\train_lora.ps1 -Char <c>` | Caption + derive repeats + train ONE LoRA. | see parameter matrix below; `-DryRun` |
| `tools\lora_train\train_all.ps1` | `train_lora` for every roster character with a dataset. | passes extra flags through |
| `python tools/lora_train/train_config.py --char <c>` | Print the resolved training params as JSON (what `train_lora` will use). | `--profile`, `--set key=value` |
| `python tools/lora_train/prep_captions.py <dir> --trigger <t>` | Prepend trigger + bake outfit tags into captions. | `--outfit`, `--prune`, `--keep`, `--dry-run`, `--strict` (fail on zero-bake) |
| `tools\lora_train\.venv\Scripts\python.exe tools\lora_train\verify_env.py` | Verify the trainer venv (GPU bf16, sd-scripts, tagger, dataset TOML parse, wildcards). | — |
| `scripts\install_qwen_edit.ps1` | Download the Qwen-Image-Edit stack. | `-Quant Q4_K_M`, `-SkipAnglesLora` |

## Training parameter matrix

Precedence (**highest wins**): explicit **CLI flag** › **`-Profile`** preset › **`[train.<char>]`** ›
**`[defaults]`**. Run `train_lora.ps1 -Char <c> -DryRun` to print the fully-resolved set + the exact
`accelerate` command, training nothing.

| CLI flag | `train.toml` key | Default | Notes |
|---|---|---|---|
| `-Dim` | `dim` | 16 | LoRA rank; 32 for visually complex characters |
| `-Alpha` | `alpha` | 8 | usually dim/2 |
| `-Optimizer` | `optimizer` | `prodigy` | `prodigy` \| `adamw` \| `adafactor` (never adamw8bit) |
| `-DCoef` | `d_coef` | 1.0 | Prodigy effective-LR knob; 0.8 reduces overcook |
| `-Steps` | `steps` | 1500 | TARGET total steps (drives `num_repeats`) |
| `-Epochs` | `epochs` | 4 | checkpoints saved, not training amount |
| `-Batch` | `batch` | 2 | raise with VRAM headroom |
| `-MinImages` | `min_images` | 12 | refuse to train on a smaller set |
| `-Lr` | `lr` | `""` | learning_rate; empty = optimizer default (adamw/adafactor 3e-4; prodigy forced 1.0) |
| `-UnetLr` | `unet_lr` | `""` | empty = 3e-4 (adamw/adafactor) |
| `-TextEncoderLr` | `text_encoder_lr` | `""` | empty = 3e-5 (adamw/adafactor) |
| `-Scheduler` | `lr_scheduler` | `cosine` | |
| `-MinSnr` | `min_snr_gamma` | `5` | |
| `-Resolution` | `resolution` | 1024 | dataset/bucket base resolution |
| `-BucketMin` | `min_bucket_reso` | 768 | |
| `-BucketMax` | `max_bucket_reso` | 1280 | |
| `-SavePrecision` | `save_precision` | `bf16` | |
| `-SaveEveryNEpochs` | `save_every_n_epochs` | 1 | |
| `-TrainTextEncoder` | `train_text_encoder` | false | also train the TE (drops `--network_train_unet_only`) |

**`[defaults]`-only keys (no CLI flag):** `network_module`, `mixed_precision`, `num_cpu_threads`,
`bucket_reso_steps`, and the Blackwell/16 GB safety toggles `cache_latents`, `cache_latents_to_disk`,
`sdpa`, `no_half_vae`, `gradient_checkpointing`. **Don't disable the safety toggles on an RTX 50xx /
16 GB card** unless you know why (see GOTCHAS). Deeper optimizer internals (prodigy `weight_decay` etc.)
stay assembled in `train_lora.ps1`.

**Non-param flags:** `-Char` (required; dataset folder + LoRA name), `-Trigger` / `-Outfit` / `-Prune`
(default from `roster.json`), `-Base` (checkpoint), `-SkipCaption` (reuse captions), `-Profile` (preset
name), `-DryRun` (preview only), `-Force` (train even if the outfit baked nothing — see the guard below).

### Outfit-bake guard

`train_lora` aborts if an outfit/prune was given but **nothing** matched the captions — i.e. the outfit
didn't bake into the trigger (the silent-failure mode: a green run, a wrong LoRA). Usual cause is a
**tagger-vocabulary mismatch** — the outfit must use the words WD14 actually writes (e.g. `thighhighs`,
not `stockings`; `panties`, not `thong underwear`). Open a dataset `.txt` to see the real tags. Pass
`-Force` to train anyway, or run `prep_captions … --dry-run` to preview what would prune.

## Profiles

Named bundles in `train.toml [profiles.<name>]`, chosen with `-Profile <name>`. Add your own table.

| Profile | Sets | For |
|---|---|---|
| `fast` | steps 800, dim 8, alpha 4 | quick, lower-quality test pass |
| `quality` | dim 16, alpha 8, steps 1500 | the default baseline, named for clarity |
| `complex` | dim 32, alpha 16, steps 2000, d_coef 0.9 | visually complex character (elaborate outfit/markings) |

```powershell
.\tools\lora_train\train_lora.ps1 -Char nyx -Profile complex          # use a preset
.\tools\lora_train\train_lora.ps1 -Char nyx -Profile complex -Steps 2400  # preset, one knob overridden
```

Per-character persistent overrides go in `train.toml`:
```toml
[train.nyx]
dim = 32
alpha = 16
```

## After training: pick the best epoch (`IL_XYPlot`)

Training saves one LoRA per epoch (`<char>_v1-0000NN` + the final `<char>_v1`). None is guaranteed
best — the **`IL_XYPlot`** workflow renders the same prompt/seed across a grid of **epoch × LoRA
strength** so you can eyeball the winner in one queue:

1. Copy the epochs to compare into `models/loras/_xyplot/` (e.g. `ursa_v1.safetensors` + a few `-0000NN`).
2. Open `IL_XYPlot`, add the trigger word (`ursachar`) to the Efficient Loader's positive prompt.
3. Keep the seed fixed; the Y axis sweeps strength 0.5→0.9 (3 columns by default). Queue once → `output/xyplot/`.
4. Keep the `(epoch, strength)` cell with the best likeness that isn't over-cooked. Load that file at
   that strength in any IL graph's LoRA bank.

Built on the **efficiency-nodes** pack. The LoRA-batch path is relative to the ComfyUI folder; set an
absolute path if your epochs live elsewhere.

## Validation guardrail

`python tools/build_il_graphs.py` validates every graph it writes:
- **Hard fail** (build exits non-zero): a rule violation (CLIP skip ≠ −2, CFG below the floor on any
  sampler) or a missing wildcard `__token__`.
- **Warning only:** missing model files (a fresh checkout won't have the multi-GB downloads).

The CFG floor is real: the validator checks `KSampler` / `KSamplerAdvanced` / `UltimateSDUpscale` /
`FaceDetailer` / `SEGSDetailer` / `CFGGuider` cfg, not just `CFGGuider`. `IL_LCM` and the
`IL_DatasetEdit_*` graphs run a low CFG by design and carry no `min_cfg` rule.

## Model compatibility (using a different base)

This toolchain is **SDXL-locked and anime/booru-flavored** — not Illustrious-only. Two separate layers:

- **Trainer = SDXL.** `train_lora.ps1` runs kohya's `sdxl_train_network.py`, which trains a LoRA for
  **any SDXL checkpoint**. Point `-Base` at another one:
  ```powershell
  .\tools\lora_train\train_lora.ps1 -Char aria -Base models\checkpoints\someOtherSDXL.safetensors
  ```
  Resolution/bucketing/dim/alpha/optimizer are generic SDXL settings, not Illustrious-specific.
- **Conventions = anime/booru.** CLIP skip −2 (validator-enforced), WD14/danbooru captioning, the
  `masterpiece, 1girl, …` prompts + embeddings, and CFG 5 suit anime SDXL models. They run on any SDXL
  but aren't right for photoreal.

| Target base model | Works? | What to do |
|---|---|---|
| Another **anime SDXL** (Pony, NoobAI, Animagine, …) | ✅ | `-Base <ckpt>` for training; swap the Checkpoint node in the render graphs. |
| **Realistic / photoreal SDXL** | ⚠️ runs, but | swap the captioner (WD14 → BLIP/natural language), edit the baked prompts + negative embeddings, and reconsider CLIP skip (−1) — change `clip_skip` in `build.py`'s rules + the `CLIPSetLastLayer` widgets. |
| **SD 1.5 / Flux / SD3 / Qwen (as the trained model)** | ❌ | Different architecture: needs a different sd-scripts train script (`train_network.py` / `flux_train_network.py` / …) + params. Not wired up. |

The dataset generator (`IL_DatasetEdit`) is the same flavor: it renders the Stage-1 hero in your SDXL
checkpoint and captions with WD14, so it's built for anime SDXL too.

## Tests

```powershell
cd tools && python -m pytest tests/ -q
```
Covers: build emits the expected graphs + roster matches `characters.toml`; generated JSON is
byte-identical to the golden fixtures; the CFG floor fires; the caption-prune logic; the param
precedence chain; and that this doc lists every flag + profile.

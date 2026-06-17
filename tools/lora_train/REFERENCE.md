# Reference — what the LoRA toolchain offers

The single "what can I tune, and where" page. Step-by-step lives in [QUICKSTART.md](QUICKSTART.md);
the full narrative in [README.md](README.md); traps in [GOTCHAS.md](GOTCHAS.md).

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
| `python tools/lora_train/prep_captions.py <dir> --trigger <t>` | Prepend trigger + bake outfit tags into captions. | `--outfit`, `--prune`, `--keep`, `--dry-run` |
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
name), `-DryRun` (preview only).

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

## Validation guardrail

`python tools/build_il_graphs.py` validates every graph it writes:
- **Hard fail** (build exits non-zero): a rule violation (CLIP skip ≠ −2, CFG below the floor on any
  sampler) or a missing wildcard `__token__`.
- **Warning only:** missing model files (a fresh checkout won't have the multi-GB downloads).

The CFG floor is real: the validator checks `KSampler` / `KSamplerAdvanced` / `UltimateSDUpscale` /
`FaceDetailer` / `SEGSDetailer` / `CFGGuider` cfg, not just `CFGGuider`. `IL_LCM` and the
`IL_DatasetEdit_*` graphs run a low CFG by design and carry no `min_cfg` rule.

## Tests

```powershell
cd tools && python -m pytest tests/ -q
```
Covers: build emits the expected graphs + roster matches `characters.toml`; generated JSON is
byte-identical to the golden fixtures; the CFG floor fires; the caption-prune logic; the param
precedence chain; and that this doc lists every flag + profile.

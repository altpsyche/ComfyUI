# Fork guide — vsiva's ComfyUI

A fork of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (upstream remote: `Comfy-Org`)
set up as a **reproducible, engine-only** install: every required custom node and the LoRA
trainer are pinned as git submodules, and a one-shot `setup.bat` provisions the whole thing.
Workflows and models live elsewhere (see [What's NOT in this repo](#whats-not-in-this-repo)).

- **Fresh-machine setup** → [ONBOARDING.md](ONBOARDING.md) (clone with submodules, prereqs, troubleshooting).
- **This file** → the map: what every batch file / script / tool dir is, and how the fork relates to upstream.

## Quick start

```powershell
git clone --recurse-submodules git@github.com:altpsyche/ComfyUI.git
cd ComfyUI
.\setup.bat                 # provision the engine (see phases below)
.\setup.bat --with-trainer  # ...and the LoRA-training venv (optional, multi-GB)
.\run_comfy.bat             # launch ComfyUI
```

## Batch files (repo root)

| File | What it does |
|---|---|
| **`setup.bat`** | One-shot, idempotent provisioner. Flags: `--gpu <mode>` (nvidia/amd-rdna3/amd-rdna35/amd-rdna4/intel-xpu/cpu), `--skip-torch`, **`--with-trainer`** (also build the LoRA trainer venv), `--no-color`. See [phases](#what-setupbat-does). |
| **`run_comfy.bat`** | Launcher: activates `venv`, sets `PYTHONUTF8=1` + disables HF telemetry, runs `python main.py`. |
| **`install_comfy.bat`** | ⚠️ Legacy stub (hand notes only) — **superseded by `setup.bat`**. Kept for reference; don't use. |

## Scripts (`scripts/`, all called by `setup.bat`)

| Script | Phase | What it does |
|---|---|---|
| **`install_torch.ps1`** | [4/6] | Maps GPU mode → PyTorch wheel index and force-reinstalls torch+torchvision. NVIDIA: autodetects CUDA from driver (≥580→cu130, ≥555→cu128, ≥525→cu121, ≥470→cu118). Skips if torch is already CUDA-enabled. |
| **`install_node_reqs.ps1`** | [5/6] | Walks `custom_nodes/`, runs each pack's `requirements.txt` then `install.py` (Manager convention), and installs `ComfyScript` editable. Warns-and-continues on per-pack failure. |
| **`install_trainer.ps1`** | optional (`--with-trainer`) | Builds the **LoRA trainer venv** at `tools/lora_train/.venv` (uv, Python 3.11): torch **cu128** for Blackwell, sd-scripts requirements, WD14-tagger deps (onnx/onnxruntime), prodigyopt, `accelerate config default`. Idempotent. |
| **`verify.ps1`** | [6/6] | Smoke checks: torch sees GPU, ComfyScript loads, key packs present, `git submodule status` clean. Prints pinned SHAs. |

### What `setup.bat` does

`[1/6]` prereqs (python/git/ssh/gpu) · `[2/6]` `git submodule update --init --recursive` (pulls
all custom nodes **and** `tools/sd-scripts`) · `[3/6]` create/activate `venv` · `[4/6]` ComfyUI
`requirements.txt` + torch (`install_torch.ps1`) · `[5/6]` custom-node deps (`install_node_reqs.ps1`)
· `[6/6]` verify · *(optional)* trainer venv (`install_trainer.ps1`, only with `--with-trainer`).

## Two Python environments (by design)

| venv | Python | torch | For |
|---|---|---|---|
| `venv/` | system (3.10+) | autodetected (e.g. cu130) | running ComfyUI |
| `tools/lora_train/.venv/` | 3.11 (via uv) | pinned 2.7.0 **cu128** | kohya sd-scripts LoRA training |

They're separate on purpose: sd-scripts needs Python ≤3.12 and a pinned torch, so it can't share
ComfyUI's venv. Both work on the RTX 5080 (Blackwell sm_120 needs CUDA 12.8+).

## `tools/` — author tooling (fork-specific)

| Path | What it is |
|---|---|
| **`tools/il_graphs/`** | Python package that generates the **IL_\*** Illustrious/SDXL workflow family (comparison ladder IL_1–5 + feature graphs IL_IPAdapter/Pose/LCM/Dataset, each with a modular **LoRA bank**). Run `python tools/build_il_graphs.py`. Details: [IL_Graphs_README.md](user/default/workflows/IL_Graphs_README.md). |
| **`tools/build_il_graphs.py`** | Thin entrypoint shim for the `il_graphs` package. |
| **`tools/validate_workflow.py`** | Validates a generated workflow JSON against its `.rules.toml` (CLIP skip −2, CFG range, required nodes). |
| **`tools/lora_train/`** | LoRA-training kit: runbook + `dataset_*.toml`, `prep_captions.py`, `train_*.ps1`, `verify_env.py`. Full flow: [tools/lora_train/README.md](tools/lora_train/README.md). |
| **`tools/sd-scripts/`** | kohya-ss/sd-scripts **submodule** (the trainer code). Provisioned by `install_trainer.ps1`. |

## Submodules

24 submodules total: **23 under `custom_nodes/`** (pinned workflow nodes + ComfyScript — see
[ONBOARDING.md](ONBOARDING.md#what-this-repo-pins)) and **`tools/sd-scripts`** (the LoRA trainer).
`git submodule status` shows exact pinned SHAs. To bump one, see ONBOARDING's *Bumping a submodule*.

## Fork vs upstream

- Remote: `origin = git@github.com:altpsyche/ComfyUI.git`; tracks `Comfy-Org/master` upstream.
- **Fork-specific** (not upstream): `setup.bat`, `run_comfy.bat`, `scripts/*.ps1`, `tools/`,
  `ONBOARDING.md`, this file, and `.gitmodules` (the pinned node set).
- Sync upstream: `git fetch upstream && git merge upstream/master` (or the existing
  `Comfy-Org:master` merge flow), then re-run `setup.bat` to pick up new core deps.

## What's NOT in this repo

- **Workflows** (`*.json`) and **models** (checkpoints/LoRAs/VAEs/ControlNets) — gitignored; models
  never live in git. Generated IL_\* workflows land in `user/default/workflows/` (also gitignored).
- **User state** (`user/`) — local only.
- The two **venvs** and `output/` — local only.

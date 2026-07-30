# Fork guide — vsiva's ComfyUI

A fork of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (upstream remote: `Comfy-Org`)
set up as a **reproducible, engine-only** install: every required custom node and the LoRA
trainer are pinned as git submodules, and a one-shot `./dev setup` provisions the whole thing.
Workflows and models live elsewhere (see [What's NOT in this repo](#whats-not-in-this-repo)).

- **Fresh-machine setup** → [ONBOARDING.md](ONBOARDING.md) (clone with submodules, prereqs, troubleshooting).
- **This file** → the map: what the `dev` CLI, the `devtools/` package, and each tool dir is, and how the fork relates to upstream.

## Quick start

Run `./dev …` on Linux/macOS, `dev …` on Windows.

```bash
git clone --recurse-submodules git@github.com:altpsyche/ComfyUI.git
cd ComfyUI
./dev setup                 # provision the engine (see phases below)
./dev setup --with-trainer  # ...and the LoRA-training venv (optional, multi-GB)
./dev run                   # launch ComfyUI
```

## The `dev` CLI (repo root)

| Command | What it does |
|---|---|
| **`./dev setup`** | One-shot, idempotent provisioner. Flags: `--gpu <mode>` (nvidia/amd-rdna3/amd-rdna35/amd-rdna4/intel-xpu/cpu), `--skip-torch`, **`--with-trainer`** (also build the LoRA trainer venv), `--no-color`. See [phases](#what-dev-setup-does). |
| **`./dev run`** | Launcher: activates `venv`, sets `PYTHONUTF8=1` + disables HF telemetry, runs `python main.py`. |
| **`./dev verify`** | Smoke checks the install (also runs as the final phase of `./dev setup`). |

## Setup phases (`devtools/setup/`, run by `./dev setup`)

These used to be standalone PowerShell scripts; they are now phases of `./dev setup`, implemented in the **devtools/ package** (setup logic under `devtools/setup/`).

| Phase | Step | What it does |
|---|---|---|
| **torch install** | [4/6] | Maps GPU mode → PyTorch wheel index and force-reinstalls torch+torchvision. NVIDIA: autodetects CUDA from driver (≥580→cu130, ≥555→cu128, ≥525→cu121, ≥470→cu118). Skips if torch is already CUDA-enabled. |
| **custom-node deps** | [5/6] | Walks `custom_nodes/`, runs each pack's `requirements.txt` then `install.py` (Manager convention), and installs `ComfyScript` editable. Warns-and-continues on per-pack failure. |
| **trainer venv** | optional (`--with-trainer`) | Builds the **LoRA trainer venv** at `tools/lora_train/.venv` (uv, Python 3.11): torch **cu128** for Blackwell, sd-scripts requirements, WD14-tagger deps (onnx/onnxruntime), prodigyopt, `accelerate config default`. Idempotent. |
| **verify** (`./dev verify`) | [6/6] | Smoke checks: torch sees GPU, ComfyScript loads, key packs present, `git submodule status` clean. Prints pinned SHAs. |

### What `./dev setup` does

`[1/6]` prereqs (python/git/ssh/gpu) · `[2/6]` `git submodule update --init --recursive` (pulls
all custom nodes **and** `tools/sd-scripts`) · `[3/6]` create/activate `venv` · `[4/6]` ComfyUI
`requirements.txt` + torch · `[5/6]` custom-node deps
· `[6/6]` verify · *(optional)* trainer venv (only with `--with-trainer`). All phases live in `devtools/setup/`.

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
| **`tools/lora_train/`** | LoRA-training kit: runbook + `./dev train <name>` (auto-captions + trains) + `./dev train --all` (whole roster) + `prep_captions.py` + `verify_env.py`. Characters come from the `CHARACTERS` roster in `il_graphs/config.py` (one `IL_DatasetEdit_<name>` Qwen-Image-Edit graph each). Full flow: [tools/lora_train/README.md](tools/lora_train/README.md). |
| **`tools/sd-scripts/`** | kohya-ss/sd-scripts **submodule** (the trainer code). Provisioned by `./dev setup --with-trainer`. |

## Submodules

32 submodules total: **31 under `custom_nodes/`** (pinned workflow nodes + ComfyScript — see
[ONBOARDING.md](ONBOARDING.md#what-this-repo-pins)) and **`tools/sd-scripts`** (the LoRA trainer).
`git submodule status` shows exact pinned SHAs. To bump one, see ONBOARDING's *Bumping a submodule*.

### 3D generation (added for the BobBlender ComfyUI track, `docs/COMFYUI.md` G0.5)

| Pack | SHA at pin | Why | Install note |
|---|---|---|---|
| **ComfyUI-TRELLIS2** (PozzettiAndrea, MIT) | `9b878516` | Microsoft TRELLIS.2-4B image-to-3D: 24 nodes, native PBR, and **open / non-manifold surfaces**, which is what makes foliage possible. Weights auto-download to `models/trellis2/` (~15 GB, bf16). | Installs through `comfy-env`, which builds an isolated pixi env at `~/.ce` with torch 2.8+cu128 and prebuilt CUDA wheels (flash-attn, sageattention, cumesh-vb, drtk, flex-gemm-ap, o-voxel-vb-ap). **That env must also carry `comfy-kitchen`**, because it imports this fork's `comfy/` source, which needs it: without it the metadata scan dies on `ModuleNotFoundError: No module named 'comfy_kitchen'` and the pack silently registers **0 nodes**. Fix: `~/.ce/.pixi/envs/trellis2-nodes/bin/python -m pip install comfy-kitchen==0.2.20` (match the version in the main venv). |
| **ComfyUI-GeometryPack** (PozzettiAndrea, MIT) | `c67199d` | Hard requirement of TRELLIS2 (`node_reqs` in its `comfy-env-root.toml`); comfy-env clones it automatically. 125 mesh nodes: load/save, decimate, remesh, UV unwrap, preview. | Pinned explicitly so a fresh clone gets it from `.gitmodules` rather than from an implicit auto-clone. |

Caveat worth knowing: comfy-env **caches each node's scanned schema**, so a COMBO whose options are
a directory listing (`GeomPackLoadMesh.file_path`) goes stale as soon as a new file lands in
`input/3d/` and a restart does not refresh it. Use `GeomPackLoadMeshPath`, which takes a free-form
path, when driving these graphs from the API.

### Controllable 3D from a block-out (added for the BobBlender ComfyUI track, `docs/COMFYUI.md` G4c)

| Pack | SHA at pin | Why | Install note |
|---|---|---|---|
| **ComfyUI-Hy3D-Omni** (Rizzlord, **no license file**) | `e513cd08` | Tencent Hunyuan3D-Omni: generation conditioned on a **point cloud, voxel volume, bounding box or skeleton**, which has no TRELLIS.2 equivalent and is what lets a Blender block-out proxy decide an asset's silhouette. Five nodes (`Hy3DOmniLoadPipeline`, `…PointGenerate`, `…VoxelGenerate`, `…BBoxGenerate`, `…PoseGenerate`). Its `control_mesh` and output are TRELLIS.2's `TRIMESH` type, so `Trellis2LoadMesh` and `Trellis2ExportTrimesh` are the mesh IO and no second wrapper is needed. | See below. It is the least maintained pack in this fork by a wide margin and it needs one weight-side fix to work at all. |

Four things to know, all measured on 2026-07-26 against torch 2.13.0+cu130 / Python 3.12 / sm_120.

1. **The control signal is silently random until you fix the checkpoint.** The pack vendors a copy of
   Tencent's `hy3dshape` in which `OmniEncoder.linear` — the MLP that projects the Fourier-embedded
   control into the DiT's token stream — has been renamed `self.liner`. The released checkpoint stores
   it as `linear.*` and the pipeline loads with `strict=False`, so the three tensors go missing, the
   projection keeps its random initialisation, and generation runs to completion with the control
   reduced to noise. Only the log says so: `442 missing and 3 unexpected keys`,
   `Missing Keys: Counter({'image_encoder': 439, 'liner': 3})`. Measured on a block-out whose shape
   nothing in the image suggests: voxel IoU against the control **0.010** before the fix, **0.53**
   after. Fix, idempotent and reversible, from the BobBlender repo:
   `venv/bin/python tools/scripts/comfy_omni_fix.py --comfy .` (it renames the three keys in the
   direction the INSTALLED wrapper needs, keeps the original as `pytorch_model.bin.orig`, and
   `--check` reports without writing). The 439 `image_encoder` keys are benign: DINOv2 comes from the
   transformers hub.
2. **The dependency list is a trap and almost none of it is needed.** Upstream's `requirements.txt`
   pins `numpy==1.24.4`, `torchaudio==2.5.1+cu124`, `deepspeed`, `open3d`, `realesrgan` and
   `pytorch-lightning==1.9.5`; installing it would destroy this venv. What the shape-only pipeline
   actually imports is five additive pure-Python packages plus one binary wheel:
   `pip install diffusers peft pytorch-lightning torchdiffeq pymeshlab` (a dry run confirms it does
   not touch torch or numpy). `torch_cluster` and `diso` are NOT needed: `torch_cluster.fps` is a
   lazy import reached only by the shape VAE's surface encoder, which the point, voxel and bbox paths
   never enter, and `diso` only by `mc_mode='dmc'`. `torch_cluster` 1.6.3 does build from source
   against this torch (`FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation
   torch_cluster`, nvcc 13.3), so it is available if a future route wants it.
3. **Weights are 13.5 GB of `.bin`, not safetensors, and there is no auto-download worth using.**
   `Hy3DOmniLoadPipeline`'s default `repo_or_path` is the HuggingFace repo id, which lands in
   `~/.cache/hy3dgen` rather than in `models/`. Pull it where the rest of the models live:
   `venv/bin/python -c "from huggingface_hub import snapshot_download;
   snapshot_download('tencent/Hunyuan3D-Omni', local_dir='models/hunyuan3d-omni',
   ignore_patterns=['*_ema.bin','assets/*'])"`. Skipping the EMA variant halves the download.
4. **It caches the pipeline in a module-level dict, so `force_reload` OOMs.** The new pipeline is
   built before the old one is dropped, i.e. two 12 GB checkpoints at once on a 16 GB card. Restart
   the server instead. The same dict is why `POST /free` cannot reclaim Omni's VRAM.

License: the repo has no `LICENSE` file, so it is all-rights-reserved as published; it is pinned as a
submodule (a pointer, not a copy) and nothing from it is redistributed. The weights carry Tencent's
own community licence, in `models/hunyuan3d-omni/License.txt` and `Notice.txt`.

### Seamless textures (added for the BobBlender ComfyUI track, `docs/COMFYUI.md` G1)

| Pack | SHA at pin | Why | Install note |
|---|---|---|---|
| **ComfyUI-seamless-tiling** (spinagon, **GPL-3.0**) | `9225ed5` | Genuinely tileable texture generation for the BobBlender track-A workflows. `SeamlessTile` switches every `Conv2d` in the UNet to circular padding and `MakeCircularVAE` does the same to the VAE decoder. Measured on one seed and prompt: an untreated wrap seam is **3.9x** the interior pixel difference, UNet-only is still **2.7x** because the VAE reintroduces it, and both together land at **0.83x**, indistinguishable from an arbitrary interior line. | Pure Python, no dependencies, nothing to provision. **Do not use its `CircularVAEDecode` node.** It `copy.deepcopy`s the live VAE per execution and discards the copy, and the server takes a **segfault inside `comfy/model_management.py load_models_gpu`** on the next decode of the session (reproduced twice, one deepcopy tolerated per server start). `MakeCircularVAE` with `copy_vae = "Make a copy"` feeding the stock `VAEDecode` gives a byte-identical image and survives repeated runs, because its copy is retained in the execution cache. |

Note this is the fork's first GPL-3.0 pinned pack rather than MIT. Harmless: ComfyUI itself is
GPL-3.0 and BobBlenderTools ships as `GPL-3.0-or-later`, and no node code is redistributed either
way. Recorded so nobody has to re-derive it.

## Fork vs upstream

- Remote: `origin = git@github.com:altpsyche/ComfyUI.git`; tracks `Comfy-Org/master` upstream.
- **Fork-specific** (not upstream): the `dev` CLI, the `devtools/` package, `tools/`,
  `ONBOARDING.md`, this file, and `.gitmodules` (the pinned node set).
- Sync upstream: `git fetch upstream && git merge upstream/master` (or the existing
  `Comfy-Org:master` merge flow), then re-run `./dev setup` to pick up new core deps.

## What's NOT in this repo

- **Workflows** (`*.json`) and **models** (checkpoints/LoRAs/VAEs/ControlNets) — gitignored; models
  never live in git. Generated IL_\* workflows land in `user/default/workflows/` (also gitignored).
- **User state** (`user/`) — local only.
- The two **venvs** and `output/` — local only.

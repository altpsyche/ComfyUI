# Onboarding (vsiva's fork)

This fork pins every required custom node + tooling submodule for reproducible engine setup across machines. **Workflows + models live in separate repos**; this repo is the engine only.

> Map of the fork (the `dev` CLI, the `devtools/` package, and every tool dir): [FORK_README.md](FORK_README.md).

Run `./dev …` on Linux/macOS, `dev …` on Windows.

```bash
# 1. Clone with all submodules (uses SSH — github SSH key required)
git clone --recurse-submodules git@github.com:<your-user>/ComfyUI.git
cd ComfyUI

# 2. One-shot engine install (verifies prereqs, inits submodules, creates venv,
#    installs torch/ComfyUI/all custom-node pip deps, runs smoke check)
./dev setup

#    Non-NVIDIA GPU? pick a mode:
./dev setup --gpu amd-rdna3      # AMD RX 7000 (Windows)
./dev setup --gpu amd-rdna35     # AMD Strix halo / Ryzen AI Max+
./dev setup --gpu amd-rdna4      # AMD RX 9000
./dev setup --gpu intel-xpu      # Intel Arc
./dev setup --gpu cpu            # CPU only

#    Optional: also build the LoRA-training venv (kohya sd-scripts, multi-GB torch cu128):
./dev setup --with-trainer       # see tools/lora_train/README.md

# 3. Get your workflows + model manifest (separate repo, planned)
#    git clone git@github.com:<your-user>/comfyui-workflows.git
#    Place per the workflows repo's own README.

# 4. Launch
./dev run
```

## Prerequisites

| Tool | Min version | Why |
|---|---|---|
| Python | 3.10 | ComfyUI core |
| Git | 2.30 | Submodule support |
| SSH key authorized for github.com | — | All submodules use `git@github.com:` URLs |
| NVIDIA GPU + driver 525+ | optional | CUDA 12.1 wheel autodetected; falls back to CPU |

If `ssh -T git@github.com` doesn't succeed, set up an SSH key first per [GitHub docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

## What this repo pins

**23 submodules** under `custom_nodes/`:

- **Workflow custom nodes** (22): ComfyUI-Manager, ComfyUI-Impact-Pack, ComfyUI-Impact-Subpack, ComfyUI_IPAdapter_plus, comfyui_controlnet_aux, ComfyUI-Advanced-ControlNet, ComfyUI_Noise, Comfy-WaveSpeed, comfyui-aesthetic-predictor-v2-5, ComfyUI-Detail-Daemon, ComfyUI-Inspire-Pack, ComfyUI-Olm-Sketch, ComfyUI-Prompt-DB, ComfyUI-ultimate-openpose-editor, sd-perturbed-attention, comfyui-sam2, comfyui-textonsegs, efficiency-nodes-comfyui, efficiency-nodes-ED, flowmatching-inverter, rgthree-comfy, was-ns
- **Author tooling** (1): ComfyScript — Python DSL for code-based workflow authoring

All pinned to specific commit SHAs. `git submodule status` shows exact versions.

## What's NOT in this repo

- **Workflows** (`MainGraphv*.json`, `workflows-src/` for v9 code authoring) — separate repo
- **Models** (checkpoints, LoRAs, VAEs, ControlNets, embeddings, etc.) — never in git; download per the workflows repo's `models/MANIFEST.toml`
- **User state** (`user/`) — local-only, gitignored

## Bumping a submodule

```sh
cd custom_nodes/<pack-name>
git fetch
git checkout <new-sha>
cd ../..
git add custom_nodes\<pack-name>
git commit -m "bump <pack> to <sha>"
```

After bumping, re-run `./dev setup` to pick up new pip deps the upstream may have added.

## Re-running setup

`./dev setup` is idempotent. Safe to re-run after pulling new commits, bumping submodules, or adding custom nodes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Phase 1 fails on SSH | Configure SSH key for github.com, retry |
| Phase 2 fails on submodule | Check upstream URL still reachable, retry with `git submodule update --init --recursive --depth 1` |
| Torch install picks wrong CUDA | Adjust the CUDA→wheel mapping in the `devtools/` package (`devtools/setup/`), re-run `./dev setup` |
| Custom node fails to load in ComfyUI | Re-run `./dev setup` (re-runs the custom-node deps phase) |
| `ComfyScript` import fails | Re-run `pip install -e custom_nodes/ComfyScript[default]` inside venv |
| `ModuleNotFoundError: comfy_aimdo` | `pip install -r requirements.txt` (`./dev setup` does this automatically) |

## Restoring a known-good state

ComfyUI-Manager snapshots (`user/__manager/snapshots/`) capture exact pack versions + python deps. If a submodule bump breaks things, restore from the most recent snapshot via Manager UI.

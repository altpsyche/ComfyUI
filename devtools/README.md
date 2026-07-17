# devtools — the ComfyUI dev toolkit

One cross-platform Python CLI for everything around this fork: provision the environment, launch
ComfyUI, generate workflow packs, download models, and train LoRAs. Runs identically on Linux,
macOS, and Windows. It replaces the old Windows-only `setup.bat` / `run_comfy.bat` / `scripts/*.ps1`
/ `tools/lora_train/*.ps1` — all that logic now lives here, in one place, tested.

> **Invocation:** `./dev <command>` on Linux/macOS, `dev <command>` on Windows (`dev.bat`). Same args
> everywhere. `setup.sh`/`setup.bat` and `run_comfy.sh`/`run_comfy.bat` remain as thin aliases for
> `./dev setup` and `./dev run`.

---

## 1. Requirements

Install these on the host **before** running setup (setup checks for them):

| Tool | Why | Notes |
|---|---|---|
| **git** | clone + submodules | with an SSH key for `github.com` if any submodule uses `git@` URLs |
| **Python 3.10+** | runs the toolkit + ComfyUI | the toolkit auto-builds the ComfyUI venv on **3.12** if your system python is 3.13+ (see below) |
| **[uv](https://docs.astral.sh/uv/)** | pins the venv interpreters | required for `--python`, the trainer venv, and the 3.13+ auto-downgrade |
| **NVIDIA driver** (optional) | GPU generation/training | setup maps the driver version to the right CUDA wheel; CPU-only also works |

Cross-platform note: the toolkit is the *only* place that knows Windows `Scripts\*.exe` vs POSIX
`bin/*`, so nothing else needs per-OS handling.

---

## 2. Install

From a fresh checkout:

```sh
./dev setup                     # NVIDIA autodetect (default)
```

That runs six idempotent phases (safe to re-run):

1. **Prereqs** — python, git, SSH to github, GPU report.
2. **Submodules** — `git submodule update --init --recursive`.
3. **venv** — creates `venv/` (the `main` venv). Detects and recreates a foreign-OS venv (e.g. a
   Windows `venv/` carried onto Linux).
4. **Core reqs + torch** — installs `requirements.txt`, then torch/torchvision for your GPU.
5. **Custom-node reqs** — every `custom_nodes/*/requirements.txt` + `install.py`, plus ComfyScript.
6. **Verify** — torch-sees-GPU, ComfyScript loads, key packs present, submodules clean.

### Options

| Flag | Effect |
|---|---|
| `--gpu <mode>` | `nvidia` (default, autodetects cuXXX from the driver) · `amd-rdna3` · `amd-rdna35` · `amd-rdna4` · `intel-xpu` · `cpu` |
| `--python <ver>` | build the `main` venv with this Python via uv (e.g. `--python 3.12`). Default: system python, **or 3.12 if system python is 3.13+** (the ML stack often has no wheels yet for brand-new Pythons) |
| `--skip-torch` | leave torch as-is (already installed) |
| `--with-trainer` | also build the **trainer** venv (kohya sd-scripts; see §6) |
| `--no-color` | plain output |

```sh
./dev setup --gpu cpu --python 3.12          # CPU-only, pinned interpreter
./dev setup --with-trainer                   # also provision LoRA training
```

---

## 3. Run

```sh
./dev run                       # launches ComfyUI → http://127.0.0.1:8188
```

Sets `PYTHONUTF8=1` + `HF_HUB_DISABLE_TELEMETRY=1` and execs the main venv's python on `main.py`.
Any extra args pass through to `main.py` (e.g. `./dev run --listen 0.0.0.0 --port 8000`). Stop with
`Ctrl-C`.

Sanity check anytime:

```sh
./dev verify
```

---

## 4. Command reference

| Command | Purpose |
|---|---|
| `./dev setup [opts]` | Provision the environment (§2). |
| `./dev run [main.py args]` | Launch ComfyUI in the main venv (§3). |
| `./dev verify` | Post-install smoke checks. |
| `./dev build [<pack>]` | Generate a pack's ComfyUI workflows (default `il_graphs`; `all` = every pack). |
| `./dev models install <pack> [--variant g=v] [--with-optional]` | Download a pack's model stack from its `models.toml`. |
| `./dev train <char> [flags]` / `./dev train --all` | Train a character LoRA. Full flag matrix: [`tools/lora_train/REFERENCE.md`](../tools/lora_train/REFERENCE.md). |
| `./dev validate <workflow.json>` | Validate a workflow against its rules (CLIP skip, CFG floor, required nodes, wildcards). |

### Examples

```sh
# Workflows
./dev build il_graphs                                  # regenerate all IL_* graphs + roster.json (+ validate)
./dev validate user/default/workflows/IL_1_Base.json

# Models (variant-aware, idempotent, reuses the HF cache)
./dev models install il_graphs                         # default quant (Q5_K_M)
./dev models install il_graphs --variant quant=Q4_K_M  # less VRAM

# Training (needs `./dev setup --with-trainer` first)
./dev train aria                                       # roster supplies trigger/outfit/prune
./dev train aria --dry-run                             # print resolved params + accelerate cmd + TOML, train nothing
./dev train nyx --profile complex --steps 2400         # preset + one override
./dev train --all                                      # every roster character with a dataset
```

---

## 5. The character-LoRA pipeline (end to end)

The two shipped tool areas — `il_graphs` (workflow generation) and the LoRA trainer — chain like this:

```
edit characters.toml ─▶ ./dev build il_graphs ─▶ roster.json + IL_DatasetEdit_<char> graphs
        │                                                    │
        │                                          ./dev models install il_graphs   (Qwen-Edit stack)
        ▼                                                    ▼
   ./dev setup --with-trainer          open IL_DatasetEdit_<char> in ComfyUI ─▶ output/dataset/<char>/
        │                                                    │
        └──────────────▶ ./dev train <char> ◀───────────────┘  (captions + trains) ─▶ models/loras/<char>_v1
```

Deeper guides: [`tools/lora_train/QUICKSTART.md`](../tools/lora_train/QUICKSTART.md) (fast loop),
[`ADD_CHARACTER.md`](../tools/lora_train/ADD_CHARACTER.md) (start to finish),
[`DATASET.md`](../tools/lora_train/DATASET.md) (the dataset engine),
[`REFERENCE.md`](../tools/lora_train/REFERENCE.md) (every flag),
[`GOTCHAS.md`](../tools/lora_train/GOTCHAS.md),
and [`tools/il_graphs/ARCHITECTURE.md`](../tools/il_graphs/ARCHITECTURE.md).

---

## 6. How it's built

- **Repo-root package**, run `python -m devtools`. `dev`/`dev.bat` just `cd` to the repo and invoke it
  on the system python.
- **Two-tier execution.** The dispatcher (`cli.py`), `core/`, and the `setup/` path import **only the
  standard library**, because `dev setup` runs on a bare system python *before* any venv exists.
  Anything that needs torch/deps (`run`, `verify`, `train`) **re-execs into the right venv**. The
  boundary is enforced by `tools/tests/test_stdlib_only.py`.
- **Named venvs** (`core/venv.py`):
  - `main` = repo `venv/` — the ComfyUI runtime.
  - `trainer` = `tools/lora_train/.venv` — uv-pinned **Python 3.11** + cu128 torch, because kohya
    sd-scripts needs Python ≤3.12 and a pinned torch. Built by `./dev setup --with-trainer`.
  Future packs can register their own. `core/platform.py` is the only place that knows `Scripts\*.exe`
  vs `bin/*`.

```
devtools/
  cli.py            dispatcher (stdlib)
  core/             platform · config · venv · download · nodes   (stdlib)
  setup/            setup · torch · node_reqs · trainer · verify
  packs/            registry · base (the Pack contract) · <pack> adapters
  train/            LoRA training (shared machinery) + cmd.py (pure, tested)
```

Generators (`il_graphs`, `lora_train`) stay under `tools/`; `devtools/packs/*` are thin adapters that
register them. Tests: `python -m pytest tools/tests -q` (or `uv run --with pytest pytest tools/tests -q`).

---

## 7. Adding a new pack

A **pack** is a self-contained tool area (anime characters, realistic characters, 3D, textures, …).
Everything that differs per pack is data in a `pack.toml`; the only code you write is how to generate
the pack's ComfyUI workflows. To add one — e.g. a `textures` pack:

1. **Create the pack's home** (keep generators under `tools/`, alongside `il_graphs`):
   `tools/textures/` with your generator code and a `pack.toml`:
   ```toml
   name = "textures"
   kind = "texture"                       # image-char | realistic-char | 3d | texture
   output_subdir = "workflows/textures"   # own namespace so packs don't collide
   schema_version = 1
   models_manifest = "models.toml"
   custom_nodes = ["SomeTextureNode"]     # names of required submodules; verify/build check presence
                                          # (git pins the SHA — no need to list it here)
   # [train]  ← omit entirely if the pack doesn't train LoRAs (3D/textures usually don't).
   ```

2. **Declare its models** in `tools/textures/models.toml` (same schema as
   [`tools/il_graphs/models.toml`](../tools/il_graphs/models.toml): `[[models]]` rows + optional
   `[[variants]]` groups). `./dev models install textures` fetches them, reusing the HF cache for
   anything shared with another pack.

3. **Write the build adapter** `devtools/packs/textures.py`:
   ```python
   from ..core import config
   from .base import Context, Pack

   class TexturesPack(Pack):
       toml = config.TOOLS / "textures" / "pack.toml"

       def build(self, ctx: Context) -> int:
           # generate JSON into ctx.output_dir / self.meta.output_subdir
           ...
           return 0

   PACK = TexturesPack
   ```

4. **Register it** in `devtools/packs/__init__.py`:
   ```python
   REGISTRY = {
       "il_graphs": "devtools.packs.il_graphs",
       "textures":  "devtools.packs.textures",
   }
   ```

That's it — `./dev build textures`, `./dev models install textures`, and (if it declares `[train]`)
`./dev train <name> --pack textures` all work, and `./dev verify` checks its custom nodes. No launcher
or dispatcher changes needed.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `python not found` from `./dev` | install Python 3.10+ and put it on PATH. |
| torch/custom-node wheels fail to install | your system python is likely too new — re-run `./dev setup --python 3.12` (needs uv). The default already auto-picks 3.12 when system python is 3.13+. |
| `venv/ is a Windows venv — recreating` | expected on a repo moved from Windows; setup rebuilds it for this OS. |
| a custom node's `install.py` fails (e.g. `No module named 'pkg_resources'`) | non-fatal — its pip deps still installed. If that node needs its bootstrap, `venv/bin/python -m pip install setuptools importlib_metadata` then re-run `venv/bin/python custom_nodes/<node>/install.py`. |
| `torch install: cpu (nvidia-smi missing)` | no NVIDIA driver detected; install the driver, or use `--gpu cpu` deliberately. |
| trainer commands say "trainer venv missing" | run `./dev setup --with-trainer` (needs uv). |
| wrong CUDA wheel | `./dev setup --gpu nvidia` maps the driver version → cuXXX; override the whole mode with `--gpu`. |

---

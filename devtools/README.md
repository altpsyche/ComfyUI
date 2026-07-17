# devtools — the cross-platform dev toolkit

One Python CLI for provisioning the env, launching ComfyUI, generating workflow packs, downloading
models, and training LoRAs. Runs identically on Linux, macOS, and Windows. Replaces the old
`setup.bat` / `run_comfy.bat` / `scripts/*.ps1` / `tools/lora_train/*.ps1` scripts — all that logic
now lives here, in one place, testable.

## Using it

```
./dev <command>          # Linux/macOS   (dev.bat <command> on Windows — same args)
```

| Command | What it does |
|---|---|
| `./dev setup [--gpu nvidia] [--skip-torch] [--with-trainer] [--no-color]` | Provision: prereqs → submodules → venv → core reqs → torch → custom-node deps → (trainer) → verify. Idempotent. |
| `./dev run` | Launch ComfyUI in the main venv. |
| `./dev verify` | Post-install smoke checks (torch GPU, ComfyScript load, packs present, submodules clean). |
| `./dev build [<pack>]` | Generate a pack's ComfyUI workflows (default `il_graphs`; `all` builds every pack). |
| `./dev models install <pack> [--variant group=value] [--with-optional]` | Download a pack's model stack from its `models.toml`. |
| `./dev train <char> [flags]` / `./dev train --all` | Train a character LoRA. See `tools/lora_train/REFERENCE.md`. |
| `./dev validate <workflow.json>` | Validate a workflow against its rules. |

`setup.sh`/`setup.bat` and `run_comfy.sh`/`run_comfy.bat` remain as thin aliases for
`./dev setup` and `./dev run`.

## How it's built

- **Repo-root package**, run `python -m devtools`. The `dev`/`dev.bat` launchers just `cd` to the repo
  and invoke it on the system python.
- **Two-tier execution.** The dispatcher (`cli.py`), `core/`, and the `setup/` path import **only the
  standard library**, because `dev setup` runs on a bare system python *before* any venv exists. Anything
  that needs torch/deps (`run`, `verify`, `train`) **re-execs into the right venv**. The boundary is
  enforced by `tools/tests/test_stdlib_only.py`.
- **Named venvs** (`core/venv.py`): `main` = repo `venv/` (ComfyUI runtime), `trainer` =
  `tools/lora_train/.venv` (uv-pinned py3.11 + cu128 torch for kohya sd-scripts). `core/platform.py` is
  the *only* place that knows `Scripts\*.exe` (Windows) vs `bin/*` (POSIX).

```
devtools/
  cli.py            dispatcher (stdlib)
  core/             platform · config · venv · download · nodes   (stdlib)
  setup/            setup · torch · node_reqs · trainer · verify
  packs/            registry · base (Pack contract) · <pack> adapters
  train/            LoRA training (shared machinery) + cmd.py (pure, tested)
```

## Adding a new pack

A **pack** is a self-contained tool area (anime characters, realistic characters, 3D, textures, …).
Everything that differs per pack is data in a `pack.toml`; the only code you write is how to generate
the pack's ComfyUI workflows. To add one — e.g. a `textures` pack:

1. **Create the pack's home** (keep generators under `tools/` alongside `il_graphs`):
   `tools/textures/` with your generator code and a `pack.toml`:
   ```toml
   name = "textures"
   kind = "texture"                 # image-char | realistic-char | 3d | texture
   output_subdir = "workflows/textures"   # own namespace so packs don't collide
   schema_version = 1
   models_manifest = "models.toml"
   custom_nodes = ["SomeTextureNode"]     # names of required submodules; verify/build check presence
                                          # (git pins the SHA — no need to list it here)
   # [train]  ← omit entirely if the pack doesn't train LoRAs (3D/textures usually don't).
   ```

2. **Declare its models** in `tools/textures/models.toml` (same schema as
   `tools/il_graphs/models.toml`: `[[models]]` rows + optional `[[variants]]` groups). `./dev models
   install textures` then fetches them, reusing the HF cache for anything shared with another pack.

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
`./dev train <name> --pack textures` all work, and `./dev verify` checks its pinned custom nodes. No
launcher or dispatcher changes needed.

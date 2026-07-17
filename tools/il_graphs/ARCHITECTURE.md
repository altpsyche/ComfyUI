# `il_graphs` — generator architecture (extend it here)

The code that emits the IL_* ComfyUI workflows. **Edit the Python here, never the generated
`user/default/workflows/*.json|*.md|*.rules.toml`** — they're overwritten on every build. The roster
itself is *data* — [`characters.toml`](characters.toml), not code.

> Using the workflows: [../../user/default/workflows/IL_Graphs_README.md](../../user/default/workflows/IL_Graphs_README.md).
> Training side: [../lora_train/README.md](../lora_train/README.md). Dataset engine internals:
> [../lora_train/DATASET.md](../lora_train/DATASET.md). Traps: [../lora_train/GOTCHAS.md](../lora_train/GOTCHAS.md).

## The loop (the contract)

```sh
./dev build il_graphs                      # regenerate all graphs + roster.json (+ validate)
python -m pytest tools/tests -q            # must stay green
```
Every emitted graph is **golden-locked**: `tests/test_golden.py` byte-compares the JSON against
`tests/golden/`. A change that alters output **fails the test on purpose** — if the change is intended,
re-snapshot by copying the new `user/default/workflows/<name>.json` into `tests/golden/`. (`IL_XYPlot`
is excluded — it bakes a machine-specific absolute path.)

## Package map

| File | Role |
|---|---|
| `build.py` | **Orchestrator.** Builds the static graphs + one `IL_DatasetEdit_<char>` per roster entry, writes `<name>.json` / `<name>.rules.toml` / `<name>.md` + `roster.json`, prunes stale outputs, then validates every graph (hard-fail on rule/wildcard violation). Entry point: `tools/build_il_graphs.py`. |
| `config.py` | Paths (`ROOT`, `SRC`, `OUT`), the shared `SEED`, base checkpoint/VAE/upscaler/ControlNet names, the base sampler constants, the `POS`/`NEG`/`HAND_POS`/`FACE_POS` prompts, `REF_SUFFIX`, and `CHARACTERS` (loads `characters.toml`, order preserved). |
| `templates.py` | Node **templates**. Harvests one template per node `type` from `SRC` (`MainGraphv10.json`), plus hand-authored `EXTRA_TEMPLATES` for nodes absent from the harvest (the Qwen-Edit nodes, etc.). `TEMPLATES[type]` is the prototype `Builder.add` deep-copies. |
| `builder.py` | The **`Builder`** class — the only thing that touches raw graph structure. `add()` clones a template + sets id/pos/mode; `link()` wires by **slot name** (not index); `group()` draws a labelled box; `build()` topo-sorts node `order` and returns the workflow dict. |
| `layers.py` | Reusable **node-group builders** that compose onto a `Builder`: `core()` (checkpoint → LoRA bank → sampler → decode), `add_upscale()`, `add_detailers()`, `add_face_inpaint()`, `add_bg()`, `add_finish()`, `lora_wv()` (LoRA-bank widgets). |
| `graphs.py` | One **`build_*()` per graph** — assembles layers into a finished workflow. `build_base/refine/guided/studio/max/ipadapter/pose/lcm/xyplot()` (static) and `build_dataset_edit(name, identity, outfit, hero_seed)` (per character). |
| `docs.py` | The per-graph **Markdown generator** (`md(name, g)`), driven by the `DOCS` table. Writes `<name>.md` next to each workflow. Hand-written docs (README/REFERENCE/this file/…) are NOT generated here. |

## Data flow

```
characters.toml ─┐
                 ├─ config.CHARACTERS ─┐
MainGraphv10.json ── templates.TEMPLATES ─┐
                                          ▼
   graphs.build_*()  ──uses──▶  layers.*  ──uses──▶  builder.Builder
                                          │
build.py ─────────────────────────────────┼─▶ user/default/workflows/<name>.json
                                           ├─▶ <name>.rules.toml   (validator rules)
                                           ├─▶ <name>.md           (docs.md)
                                           └─▶ tools/lora_train/roster.json  (name/trigger/id/outfit/prune)
```

## The 15 graphs

**9 static** (in `build.py`'s `graphs` dict): `IL_1_Base`, `IL_2_Refine`, `IL_3_Guided`,
`IL_4_Studio`, `IL_5_Max`, `IL_IPAdapter`, `IL_Pose`, `IL_LCM`, `IL_XYPlot`.
**6 dynamic** — one `IL_DatasetEdit_<char>` per `characters.toml` table (currently aria, aria_gala,
kael, nyx, mira, mira_winter). Add/remove a character → that count changes; tests assert
`roster ↔ characters.toml` and that every expected graph emits + validates.

## Where to change what

| Want to change | Edit |
|---|---|
| Add/remove a character (data) | [`characters.toml`](characters.toml) — no code |
| Base checkpoint / VAE / sampler / seed / prompts | `config.py` |
| Hero ref framing (full-body vs portrait) | `REF_SUFFIX` in `config.py` |
| Qwen-Edit instruction / LoRA strengths / steps | `build_dataset_edit()` in `graphs.py` |
| A tier's composition (add/remove a layer) | the relevant `build_*()` in `graphs.py` |
| A reusable node group (upscale/detailer/…) | `layers.py` |
| A node type the harvest source lacks | add to `EXTRA_TEMPLATES` in `templates.py` |
| What a graph's `.md` says | the `DOCS` entry in `docs.py` |
| Validator rules emitted per graph (CLIP skip / CFG floor / required nodes) | `build.py` (`cfg_rules` / `cs_rule` / `req`) |

A missing template raises a clear `KeyError` from `Builder.add` telling you to add it to
`EXTRA_TEMPLATES`. `link()` raises if a slot name doesn't exist — wire by name, and the error lists the
available slots.

## Adding a new model family (the growth seam)

The toolchain is SDXL-locked today; extending to another family is additive, not a rewrite:

1. **Generation:** add a new `build_<family>_dataset_edit()` (or a parallel dataset builder) in
   `graphs.py`, with any new node types in `EXTRA_TEMPLATES`. Emit it from `build.py` with its own
   `*.rules.toml` (set/clear the CLIP-skip and CFG-floor rules as that model needs). Golden-snapshot it.
2. **Training:** point `./dev train --base` at the checkpoint for another **SDXL** model (already
   supported); a non-SDXL model needs a different sd-scripts script (`train_network.py` /
   `flux_train_network.py` / …) — see the **model-compatibility matrix** in
   [../lora_train/REFERENCE.md](../lora_train/REFERENCE.md).
3. **Docs:** add a `DATASET_<family>.md` for that engine and a row to the REFERENCE compatibility
   matrix. README stays the stable index — it shouldn't need rewriting.

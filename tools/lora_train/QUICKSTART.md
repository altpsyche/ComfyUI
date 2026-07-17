# Character LoRA — quick start

Terse loop. **Can't-miss checklist: [ADD_CHARACTER.md](ADD_CHARACTER.md).** Concepts + setup:
[README.md](README.md). Dataset engine: [DATASET.md](DATASET.md). Every knob: [REFERENCE.md](REFERENCE.md).

## Once per machine

Run `./dev …` on Linux/macOS, `dev …` on Windows.

```bash
./dev setup --with-trainer        # trainer venv
./dev models install il_graphs    # Qwen-Edit stack (~23 GB)
```

## New character

1. Add a table to [`tools/il_graphs/characters.toml`](../il_graphs/characters.toml) (data file — no Python):
   ```toml
   [aria]
   id = "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles"   # identity only, no clothes
   outfit = "tennis uniform, teal and white tennis dress, white visor, white shoes"  # optional, auto-locked
   ```
2. Regenerate:
   ```sh
   ./dev build il_graphs
   ```
3. In ComfyUI, open `IL_DatasetEdit_aria` (details: [DATASET.md](DATASET.md)):
   - **Stage 1:** reroll **Hero Seed** until the face in HERO preview is good. Leave it fixed.
   - **Stage 2:** seed control = randomize, batch count ~40, **Queue once** -> `output/dataset/aria/`.
4. Curate: delete bad frames in place. Keep best 25-40 (min 12).
5. Train:
   ```bash
   ./dev train aria        # or ./dev train --all for the whole roster
   ```
6. Use: any IL_* graph -> **LoRA bank** -> toggle `aria_v1` ON, strength ~0.75, add trigger `ariachar`.

Output: `models/loras/aria_v1.safetensors` + one per epoch (`-000001..`). Pick the best epoch via
XY-plot of strength {0.5, 0.75, 0.9} in IL_1_Base.

## Outfit

- Put the signature outfit in the entry's `outfit` string. It auto-bakes into the trigger at train
  time (colour/style variants too) -> renders identically in every scene. Nothing else to do.
- Leave `outfit: ""` to let the checkpoint pick clothes (promptable, not locked).

## Same character, multiple outfits

Add a `like` table per costume -> separate LoRA, same facial identity:

```toml
[aria_gala]
like = "aria"
outfit = "elegant emerald evening gown, long gloves, high heels"
```

`like` inherits `aria`'s `id` + `hero_seed` + `prune`; you write only the new `outfit`. Regenerate -> generate
`IL_DatasetEdit_aria_gala` -> train. Trigger `aria_galachar`.

**For the closest face match:** after rerolling `aria`'s Hero Seed to a face you like, write that seed
into the parent table as `hero_seed = <value>`, regenerate, *then* the variants inherit it. (Faces are
recognizably the same person, not pixel-identical -- the outfit changes the render and each is its own LoRA.)

## Common knobs

| Want | Do |
|---|---|
| Harder face lock | `./dev train aria --train-text-encoder` or `--dim 32 --alpha 16` |
| More capacity | `--dim 32 --alpha 16` (or `--profile complex`) |
| Preset bundle | `--profile fast\|quality\|complex` (defined in `train.toml`) |
| Preview the exact command, train nothing | `--dry-run` |
| Persist per-char training params | a `[train.<char>]` table in `train.toml` |
| LoRA overcooked | `--d-coef 0.8` or `--optimizer adamw` |
| OOM / too slow | `./dev models install il_graphs --variant quant=Q4_K_M` |
| More pose/angle variety | raise multiple-angles LoRA toward 1.0 in `build_dataset_edit()`; add lines to `wildcards/pose.txt` |
| Don't re-tag on retrain | `./dev train aria --skip-caption` |
| Preview caption pruning | `prep_captions.py <dir> --trigger <t> --outfit "..." --dry-run` |

All training knobs + precedence: **[REFERENCE.md](REFERENCE.md)**. Tune them in `train.toml` (no code edit).

Wildcard `.txt` edits (`custom_nodes/ComfyUI-Impact-Pack/wildcards/`) are live -- reload the graph, no
regenerate. `characters.toml` (`id`/`outfit`/`like`) edits need `python tools/build_il_graphs.py` + re-open the graph.

Steering pose/angle/scene variety: **[WILDCARDS.md](WILDCARDS.md)**.
Before changing the pipeline: **[GOTCHAS.md](GOTCHAS.md)** (dead ends + traps — don't re-do them).

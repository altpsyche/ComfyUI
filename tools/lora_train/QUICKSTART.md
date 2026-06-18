# Character LoRA — quick start

Terse loop. **Can't-miss checklist: [ADD_CHARACTER.md](ADD_CHARACTER.md).** Full detail:
[README.md](README.md). Every knob in one table: [REFERENCE.md](REFERENCE.md).

## Once per machine

```powershell
setup.bat --with-trainer                                                 # trainer venv
powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1   # Qwen-Edit stack (~23 GB)
```

## New character

1. Add a table to [`tools/il_graphs/characters.toml`](../il_graphs/characters.toml) (data file — no Python):
   ```toml
   [aria]
   id = "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles"   # identity only, no clothes
   outfit = "tennis uniform, teal and white tennis dress, white visor, white shoes"  # optional, auto-locked
   ```
2. Regenerate:
   ```powershell
   python tools/build_il_graphs.py
   ```
3. In ComfyUI, open `IL_DatasetEdit_aria`:
   - **Stage 1:** reroll **Hero Seed** until the face in HERO preview is good. Leave it fixed.
   - **Stage 2:** seed control = randomize, batch count ~40, **Queue once** -> `output/dataset/aria/`.
4. Curate: delete bad frames in place. Keep best 25-40 (min 12).
5. Train:
   ```powershell
   .\tools\lora_train\train_lora.ps1 -Char aria        # or train_all.ps1 for the whole roster
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
| Harder face lock | `train_lora.ps1 -Char aria -TrainTextEncoder` or `-Dim 32 -Alpha 16` |
| More capacity | `-Dim 32 -Alpha 16` (or `-Profile complex`) |
| Preset bundle | `-Profile fast\|quality\|complex` (defined in `train.toml`) |
| Preview the exact command, train nothing | `-DryRun` |
| Persist per-char training params | a `[train.<char>]` table in `train.toml` |
| LoRA overcooked | `-DCoef 0.8` or `-Optimizer adamw` |
| OOM / too slow | `install_qwen_edit.ps1 -Quant Q4_K_M` |
| More pose/angle variety | raise multiple-angles LoRA toward 1.0 in `build_dataset_edit()`; add lines to `wildcards/pose.txt` |
| Don't re-tag on retrain | `train_lora.ps1 -Char aria -SkipCaption` |
| Preview caption pruning | `prep_captions.py <dir> --trigger <t> --outfit "..." --dry-run` |

All training knobs + precedence: **[REFERENCE.md](REFERENCE.md)**. Tune them in `train.toml` (no code edit).

Wildcard `.txt` edits (`custom_nodes/ComfyUI-Impact-Pack/wildcards/`) are live -- reload the graph, no
regenerate. `characters.toml` (`id`/`outfit`/`like`) edits need `python tools/build_il_graphs.py` + re-open the graph.

Steering pose/angle/scene variety: **[WILDCARDS.md](WILDCARDS.md)**.
Before changing the pipeline: **[GOTCHAS.md](GOTCHAS.md)** (dead ends + traps — don't re-do them).

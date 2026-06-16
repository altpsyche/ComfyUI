# Wildcards — steering dataset variety

The `IL_DatasetEdit_<name>` edit instruction varies each frame by pulling one random line from each
wildcard file. Edit those files to control the kind of variety your dataset gets.

## Where

```
custom_nodes/ComfyUI-Impact-Pack/wildcards/*.txt
```

> These live in the Impact-Pack **submodule** (untracked by this fork). They exist on this machine;
> a fresh clone won't have them — re-create them there if you set up elsewhere.

## Format

- **One option per line.** Plain text. No commas, no `(weights:1.1)`, no quotes.
- Blank lines are ignored. Order doesn't matter (one line is picked at random per roll).
- Each `__name__` token in the instruction maps to `name.txt` (e.g. `__pose__` -> `pose.txt`).

Example `pose.txt`:
```
standing
sitting
arms crossed
looking back over shoulder
jumping
```

## Files the dataset tool uses

The edit instruction is `..., __angle__, __pose__, __expression__, __framing__, __background__, __lighting__`:

| Token | File | Steers |
|---|---|---|
| `__angle__` | `angle.txt` | camera angle (front view, from below, dutch angle, …) |
| `__pose__` | `pose.txt` | body pose / action (standing, kneeling, waving, …) |
| `__expression__` | `expression.txt` | face (smiling, neutral, surprised, …) |
| `__framing__` | `framing.txt` | shot size (full body, close-up portrait, wide shot, …) |
| `__background__` | `background.txt` | scene (city street, forest, studio backdrop, …) |
| `__lighting__` | `lighting.txt` | light (golden hour, neon glow, softbox, …) |

(`outfit.txt` and `hand_pose.txt` exist but are **not** referenced by the edit instruction — ignore them.)

## Edit + apply

1. Open the `.txt`, add/remove/change lines (one per line), save.
2. **Reload the graph** in ComfyUI (or queue again) — wildcards are read live. **No `build_il_graphs.py`
   regenerate needed.**

## Tips

- **Widen variety:** add lines (`from below`, `dutch angle`, `lying down`, `rainy street`). More lines =
  more spread across your ~40 frames.
- **Narrow it:** delete lines you don't want (e.g. drop `from behind` if back-of-head frames waste slots).
- **Keep options dataset-appropriate.** Every line should be a pose/scene a *clean training image* can
  show — avoid extreme crops or occlusions that hide the face.
- **Want a new axis?** Add the token to `wtext` in `build_dataset_edit()`
  ([`tools/il_graphs/graphs.py`](../il_graphs/graphs.py)) AND create the matching `.txt`, then regenerate.
  Keep `__angle__/__pose__` leading the instruction — see README §6.4 (Qwen moves the pose less if scene
  axes come first).
- A wildcard token printed *literally* in the image = the `.txt` is missing or misnamed; fix the path and
  reload.

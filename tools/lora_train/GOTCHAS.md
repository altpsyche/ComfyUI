# Gotchas + dead ends — read before changing things

Hard-won lessons. README §13 says why the current design is good; **this file says what NOT to undo**
and what will silently bite you. If something here looks "obviously improvable," it was probably already
tried — check before reverting.

## Don't re-do these (tried, reverted)

| Tempting change | Why it's wrong |
|---|---|
| Set Edit-instruction node to `mode: fixed` | Stops the UI from showing the resolved prompt (bottom box shows raw `__wildcards__`) and feels slower. The fix for headless was **wildcards in both text fields**, keep `populate`. |
| Lead the instruction with the change (`Change the shot to __framing__ … Re-pose …`) | Qwen is conservative at 6 steps; leading with framing/scene makes it spend the budget on zoom/bg/lighting and **move the pose less**. Keep `__angle__/__pose__` **first**; append scene axes. |
| Concrete (wildcard-free) text in `populated_text` | That was the root cause of identical headless frames — the backend expands `populated_text`, so it must hold the `__wildcards__`, not a frozen roll. |
| Bring back the hero + IPAdapter / danbooru-`base` dataset route | Deleted on purpose. IPAdapter on the whole base washed out the render; a ReActor/high-denoise face-swap **froze the expression** (monotone, bad for training). Qwen-edit is the only route now. |
| Crank `multiple-angles` LoRA to 1.0 by default | 1.0 was tried; identity drifts. **0.8** is the kept default — raise per-graph only if you need more angle push. |
| `--clip_skip` in the trainer | `sdxl_train_network.py` ignores it for SDXL (warns + no-ops). Inference-side CLIP skip -2 is unrelated. |
| `-Optimizer adamw8bit` | Needs bitsandbytes, unverified on Blackwell `sm_120`. That's why Prodigy is the default. Use `adamw`/`adafactor` to A/B, not 8bit. |
| Default to 10 epochs | Epochs only set checkpoint granularity (1 LoRA saved/epoch); `num_repeats` keeps total steps ~constant. Default is **4** — enough checkpoints to pick from without bloat. |

## Things that silently bite

- **Machine sleep / GPU TDR kills training with no traceback.** A run died at step 566/1532 — all python
  gone, GPU idle, no error. It was the machine sleeping mid-train. **Disable sleep before a long train.**
  Per-epoch checkpoints survive, so you lose at most one epoch; relaunch with `-SkipCaption`.
- **Other GPU apps slow Qwen generation, not your settings.** A RenderDoc capture holding 11 GB VRAM made
  generation crawl; it wasn't the instruction/LoRA edits (those add zero compute). Check `nvidia-smi`
  before blaming the graph.
- **Free ComfyUI's VRAM before training.** ComfyUI holds models resident. Before a train, POST
  `{"unload_models":true,"free_memory":true}` to `http://127.0.0.1:8188/free` (or close ComfyUI) so the
  trainer gets the full 16 GB.
- **Re-open the graph after every `build_il_graphs.py`.** ComfyUI caches loaded graphs; edits won't show
  until you re-open the workflow.
- **Wildcard `.txt` files are in the Impact-Pack submodule (untracked).** They exist on this machine but
  a fresh clone won't have them — re-create them if you set up elsewhere. See [WILDCARDS.md](WILDCARDS.md).
- **"Same face" across outfit variants is not pixel-identical.** A `like:` variant re-renders its own hero
  with the new outfit in the prompt, and trains a separate LoRA — faces are recognizably the same person.
  Pin `hero_seed` + a tight `id` to maximize the match. (Full note: README §5 / config.py comments.)

## Why the locked settings are what they are (don't churn without a reason)

- **GGUF Q5 + Lightning @ 6 steps / cfg 1.0** — the only way the 20B edit model fits + runs fast on 16 GB.
  Drop to `Q4_K_M` only on OOM; raise steps to 8-10 only if edits look soft.
- **Stage-1 hero in YOUR checkpoint** (not a photo, not the edit model) — keeps every edited frame on your
  art style. Importing from a photo imports the edit-model look.
- **Prodigy + `--sdpa`** — Prodigy auto-tunes LR (no bitsandbytes); `--sdpa` avoids xformers wheels on
  Blackwell. Both chosen to sidestep `sm_120` wheel pain.
- **dim 16 / alpha 8, ~1500 steps** — solid character-LoRA baseline. Bump to `-Dim 32 -Alpha 16` for
  visually complex characters; that's the first knob, not steps.

## If you change the generator

All workflow changes go through `tools/il_graphs/` (edit the Python, never the generated JSON/md) then
`python tools/build_il_graphs.py`. New custom nodes = git submodules. Validate with
`python tools/validate_workflow.py user/default/workflows/<graph>.json`. Stale/renamed dataset graphs are
pruned automatically on regenerate.

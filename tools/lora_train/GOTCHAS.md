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
  Pin `hero_seed` + a tight `id` to maximize the match. (Full note: README §5 / characters.toml comments.)
- **Training defaults live in `train.toml`, not the .ps1.** Don't expect `dim 16` / `steps 1500` hardcoded
  in `train_lora.ps1`'s `param()` block — they resolve from `train.toml [defaults]`. **Precedence (highest
  wins): explicit CLI flag > `-Profile` > `[train.<char>]` > `[defaults]`.** If a value "won't take
  effect," something higher in that chain is overriding it — run `train_lora.ps1 -Char <c> -DryRun` to
  print the resolved set + the exact `accelerate` command before committing to a multi-hour train.
- **The Blackwell/16 GB safety toggles are now exposed in `train.toml` (`sdpa`, `no_half_vae`, bf16,
  `gradient_checkpointing`, latent caching).** Exposed ≠ "tune freely" — they default ON for a reason
  (sm_120 wheel pain, black VAE tiles, 16 GB headroom). Don't flip them off without one.

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
`python tools/build_il_graphs.py`. The roster itself is data — `tools/il_graphs/characters.toml`, not
Python. New custom nodes = git submodules. **The build now validates every graph it writes** and
**hard-fails on a rule (CLIP skip ≠ −2, CFG below the floor) or a missing wildcard** (missing model
files are only a warning); skip with `--no-validate`. Standalone:
`python tools/validate_workflow.py user/default/workflows/<graph>.json`. Stale/renamed dataset graphs —
and stale `.cache/*.toml` for removed characters — are pruned automatically on regenerate.
There are tests now: `cd tools && python -m pytest tests/ -q` (golden-locks the graph JSON, so a
generator change that alters output fails until you re-snapshot on purpose).

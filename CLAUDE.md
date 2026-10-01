# CLAUDE.md — ComfyUI fork (altpsyche)

This is the engine fork: upstream ComfyUI with no core patches, every custom node pack pinned as a
submodule, and a `.gitignore` identical to upstream. Read `AGENTS.md` (upstream's node style rules)
and `FORK_README.md` before changing anything. The author tooling lives in `~/dev/comfy-devkit`.

## Sprite work

`custom_nodes/ComfyUI-SpriteKit` is a submodule whose source repo is `~/dev/ComfyUI-SpriteKit`.
All sprite work follows the devkit's rules and protocol:

- `~/dev/comfy-devkit/AGENTS.md` — the hard rules (builders stop at `[r]`, author-only actions,
  never bypass hooks).
- `~/dev/comfy-devkit/docs/SPRITE_HANDOFF.md` and `docs/SPRITE_TASKS.md` — protocol and queue.

Rules specific to this repo:

- **Never commit `.gitmodules` or the `custom_nodes/ComfyUI-SpriteKit` gitlink** without the
  author's go-ahead in the current session. Its URL is still a local path until the GitHub repo
  exists. `.githooks/pre-commit` blocks it unless the author runs it with `SPRITEKIT_PIN_OK=1`.
- Edit SpriteKit in `~/dev/ComfyUI-SpriteKit`, not in the mounted submodule; then fast-forward the
  mount (`git -C custom_nodes/ComfyUI-SpriteKit fetch && git -C … checkout <sha>`).
- No core patches, no edits to upstream files (including `AGENTS.md` and `.gitignore`), no pushes
  without the author.
- Never `--no-verify`; never edit or disable the hooks.

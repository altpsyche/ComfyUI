#!/usr/bin/env sh
# Convenience alias for `./dev run` (Linux/macOS) — launches ComfyUI in the main venv.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$here/dev" run "$@"

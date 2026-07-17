#!/usr/bin/env sh
# Convenience alias for `./dev setup` (Linux/macOS). See `./dev setup --help`.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$here/dev" setup "$@"

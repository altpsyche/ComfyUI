"""The `il_graphs` pack: Illustrious/SDXL anime-character workflow family.

Thin adapter over the existing tools/il_graphs generator (kept physically in tools/ so its
ROOT-relative paths, roster output, and the tools/tests flat imports all keep working). This
registers it as a pack and drives its build; the declarative bits live in tools/il_graphs/pack.toml.
"""
from __future__ import annotations

import sys

from ..core import config
from .base import Context, Pack


class ILGraphsPack(Pack):
    toml = config.TOOLS / "il_graphs" / "pack.toml"

    def build(self, ctx: Context) -> int:
        # il_graphs is a package under tools/; import it the way build_il_graphs.py does.
        tools = str(config.TOOLS)
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import il_graphs.build as builder

        saved = sys.argv
        sys.argv = ["build"]          # default: validate (drop --no-validate)
        try:
            builder.main()            # writes IL_*.json/.rules.toml/.md + roster.json; validates
        except SystemExit as e:       # validation failure exits non-zero
            return int(e.code or 0)
        finally:
            sys.argv = saved
        return 0


PACK = ILGraphsPack

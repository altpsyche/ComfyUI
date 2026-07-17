"""Repo-relative paths + TOML loading. STDLIB ONLY at import time.

`ROOT` is the repo root: this file is devtools/core/config.py, so parents[2] is the checkout.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
TOOLS = ROOT / "tools"
CUSTOM_NODES = ROOT / "custom_nodes"


def load_toml(path):
    """Parse a TOML file. Import is deferred so this module stays stdlib-clean at import time
    (tomllib is 3.11+; on 3.10 we fall back to the tomli backport, matching the house style)."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py3.10
        import tomli as tomllib
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))

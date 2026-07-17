"""`dev run` — launch ComfyUI in the main venv. STDLIB ONLY.

Re-execs the main venv's python on main.py (no shell activation needed). Sets the same stability
env vars the old run_comfy.bat did.
"""
from __future__ import annotations

import os

from .core import config
from .core import platform as plat
from .core import venv


def run(args) -> int:
    if not venv.exists("main"):
        plat.err("main venv not found — run: ./dev setup")
        return 1
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    extra = getattr(args, "rest", None) or []
    plat.step("launching ComfyUI ...")
    return venv.reexec("main", ["main.py", *extra], env=env, cwd=config.ROOT)

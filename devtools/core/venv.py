"""Named venv registry. STDLIB ONLY.

The repo has (at least) two venvs with different interpreters:
  main    = <repo>/venv                     ComfyUI runtime (system python)
  trainer = <repo>/tools/lora_train/.venv   kohya sd-scripts (uv-pinned py3.11 + cu128 torch)

Future packs may register their own. Callers ask this module "which python for handle X" rather
than hardcoding Scripts/ vs bin/ anywhere.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config
from . import platform as plat

VENVS = {
    "main": config.ROOT / "venv",
    "trainer": config.TOOLS / "lora_train" / ".venv",
}


def path(handle: str) -> Path:
    return VENVS[handle]


def python(handle: str) -> Path:
    return plat.venv_python(VENVS[handle])


def bin(handle: str, name: str) -> Path:  # noqa: A001 - deliberate, mirrors the concept
    return plat.venv_bin(VENVS[handle], name)


def exists(handle: str) -> bool:
    return plat.venv_python(VENVS[handle]).exists()


def is_foreign(handle: str) -> bool:
    """True if the venv dir exists but was built for the OTHER OS (e.g. a Windows venv on Linux)."""
    d = VENVS[handle]
    if not d.exists():
        return False
    has_posix = (d / "bin").is_dir()
    has_win = (d / "Scripts").is_dir()
    if plat.IS_WINDOWS:
        return has_posix and not has_win
    return has_win and not has_posix


def ensure_main(*, python_spec=None, recreate: bool = False) -> bool:
    """Create <repo>/venv. Detect + recreate a foreign-OS venv (the tracked Windows venv/ carried
    onto Linux). If `python_spec` is given, build it with that interpreter via `uv venv --seed`
    (so it still ships pip); otherwise use the current system python. Returns True on success."""
    d = VENVS["main"]
    if is_foreign("main"):
        other = "POSIX" if plat.IS_WINDOWS else "Windows"
        plat.warn(f"existing venv/ is a {other} venv — recreating for this OS")
        recreate = True
    if recreate and d.exists():
        shutil.rmtree(d)
    if exists("main"):
        plat.ok("venv already exists")
        return True
    if python_spec:
        if not plat.have("uv"):
            plat.err(f"--python {python_spec} needs uv (https://docs.astral.sh/uv/); "
                     "omit it to use the system python")
            return False
        # --seed installs pip/setuptools/wheel so the later `python -m pip` steps work.
        rc = plat.run(["uv", "venv", "--seed", "--python", python_spec, str(d)])
        src = f"uv python {python_spec}"
    else:
        rc = plat.run([sys.executable, "-m", "venv", str(d)])
        src = sys.executable
    if rc != 0:
        plat.err("venv creation failed")
        return False
    plat.ok(f"created venv ({src})")
    return True


def reexec(handle: str, args, *, env=None, cwd=None) -> int:
    """Run a venv's python with `args`. Used to hand off from the stdlib dispatcher into a venv
    that has the heavy deps (torch etc). subprocess (not execv) so it behaves the same on Windows."""
    py = python(handle)
    if not py.exists():
        plat.err(f"{handle} venv missing at {VENVS[handle]}")
        return 1
    return plat.run([py, *args], env=env, cwd=cwd)

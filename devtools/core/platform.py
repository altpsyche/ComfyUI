"""OS primitives: venv-exe resolution, subprocess wrappers, prereq checks, colored status.

This is the ONE place that knows `Scripts\\*.exe` (Windows) vs `bin/*` (POSIX). Everything else
asks here. STDLIB ONLY — imported by the dispatcher and the setup path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"

# Best-effort: turn on ANSI (VT) processing on modern Windows consoles so our color codes render.
if IS_WINDOWS:  # pragma: no cover - exercised only on Windows
    try:
        import ctypes

        _k = ctypes.windll.kernel32
        _k.SetConsoleMode(_k.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING | defaults
    except Exception:
        pass


# --------------------------------------------------------------------------- venv exe resolution
def venv_bin(venv_dir, name: str) -> Path:
    """Absolute path to an executable inside a venv, for the current OS.

    Windows: <venv>\\Scripts\\<name>.exe   POSIX: <venv>/bin/<name>
    """
    venv_dir = Path(venv_dir)
    if IS_WINDOWS:
        exe = name if name.lower().endswith(".exe") else f"{name}.exe"
        return venv_dir / "Scripts" / exe
    return venv_dir / "bin" / name


def venv_python(venv_dir) -> Path:
    return venv_bin(venv_dir, "python")


# --------------------------------------------------------------------------- process helpers
def run(cmd, *, cwd=None, env=None, check=False) -> int:
    """Run a command (list of args or Paths), streaming its output. Returns the exit code.

    No shell — args are passed through verbatim, so paths with spaces are safe.
    """
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def capture(cmd, *, cwd=None) -> tuple[int, str]:
    """Run a command and capture stdout (stderr folded in). Returns (rc, text)."""
    try:
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError:
        return 127, ""
    return proc.returncode, proc.stdout or ""


def have(name: str):
    """Path to `name` on PATH, or None."""
    return shutil.which(name)


# --------------------------------------------------------------------------- colored output
_COLOR = None  # tri-state: None = auto (tty + no NO_COLOR); True/False = forced
_CODES = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "dim": "2", "bold": "1"}


def set_color(enabled: bool) -> None:
    global _COLOR
    _COLOR = enabled


def _color_on() -> bool:
    if _COLOR is not None:
        return _COLOR
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def c(text: str, color: str) -> str:
    if not _color_on():
        return text
    return f"\033[{_CODES.get(color, '0')}m{text}\033[0m"


def heading(text: str) -> None:
    print(c(f"\n=== {text} ===", "cyan"))


def ok(msg: str) -> None:
    print(f"  {c('[+]', 'green')} {msg}")


def warn(msg: str) -> None:
    print(f"  {c('[!]', 'yellow')} {msg}")


def err(msg: str) -> None:
    print(f"  {c('[x]', 'red')} {msg}")


def info(msg: str) -> None:
    print(f"  {c('[-]', 'dim')} {msg}")


def step(msg: str) -> None:
    print(f"  {c('[>]', 'cyan')} {msg}")

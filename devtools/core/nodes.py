"""Check a pack's declared custom_nodes are present and at the pinned SHA. STDLIB ONLY.

Reuses `git submodule status` (the same source verify.ps1 used) rather than re-walking .gitmodules.
"""
from __future__ import annotations

import re

from . import config
from . import platform as plat

_LINE = re.compile(r"^([\-+U ])?([0-9a-f]{7,40})\s+(\S+)")


def submodule_shas() -> dict:
    """Map submodule path -> (sha, state) where state is ' ' clean, '-' uninitialized,
    '+' checked-out-differs, 'U' merge conflict."""
    rc, out = plat.capture(["git", "submodule", "status"], cwd=config.ROOT)
    res = {}
    if rc != 0:
        return res
    for line in out.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        state = m.group(1) if m.group(1) and m.group(1) in "+-U" else " "
        res[m.group(3)] = (m.group(2), state)
    return res


def check(declared) -> tuple[list, list]:
    """declared: iterable of node NAMES (str), or {name, sha?} dicts. Returns (missing, mismatched).

    Presence + initialized state is authoritative here; the exact pinned SHA is git's job (the
    submodule gitlink, plus `dev verify`'s global submodule-status check). A dict entry MAY carry an
    optional `sha` to also assert a specific commit, but plain names are the norm so nothing needs
    hand-syncing when a submodule is bumped. `name` matches the trailing path component
    (custom_nodes/<name>).
    """
    shas = submodule_shas()
    by_leaf = {path.rsplit("/", 1)[-1]: (sha, st) for path, (sha, st) in shas.items()}
    missing, mismatched = [], []
    for node in declared:
        name = node if isinstance(node, str) else node.get("name")
        entry = by_leaf.get(name)
        if entry is None or entry[1] == "-":
            missing.append(name)
            continue
        want = None if isinstance(node, str) else node.get("sha")
        if want and not entry[0].startswith(want[: len(entry[0])]) and not want.startswith(entry[0]):
            mismatched.append(name)
    return missing, mismatched

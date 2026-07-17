"""Provision every custom-node pack. STDLIB ONLY.

Port of scripts/install_node_reqs.ps1: for each dir under custom_nodes/, `pip install -r
requirements.txt` (if present) then run its `install.py` (the ComfyUI-Manager convention, e.g.
comfyui_text_to_pose clones its model lib). Installs ComfyScript editable last.
"""
from __future__ import annotations

import os

from ..core import config
from ..core import platform as plat

_SKIP = {"__pycache__", "ComfyScript"}


def install(py) -> int:
    cn = config.CUSTOM_NODES
    results = {}
    failed = []

    def record(pack, status):
        results[status] = results.get(status, 0) + 1

    for entry in sorted(os.scandir(cn), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in _SKIP or entry.name.startswith("."):
            continue
        pack = entry.name
        req = os.path.join(entry.path, "requirements.txt")
        inst = os.path.join(entry.path, "install.py")
        has_req, has_inst = os.path.exists(req), os.path.exists(inst)

        if not has_req and not has_inst:
            record(pack, "no-reqs")
            continue

        if has_req:
            print(f"  {plat.c('[>]', 'cyan')} {pack} (requirements.txt) ...")
            if plat.run([py, "-m", "pip", "install", "-U", "-r", req]) == 0:
                record(pack, "installed")
            else:
                record(pack, "failed")
                failed.append(pack)

        if has_inst:
            # ComfyUI-Manager runs install.py on install; mirror it. Runs from the pack dir.
            print(f"  {plat.c('[>]', 'cyan')} {pack} (install.py) ...")
            if plat.run([py, "install.py"], cwd=entry.path) == 0:
                record(pack, "install.py")
            else:
                record(pack, "failed")
                failed.append(f"{pack} (install.py)")

    # ComfyScript: editable install with [default] extras (transpiler + node import).
    cs = cn / "ComfyScript"
    if (cs / "pyproject.toml").exists():
        print(f"  {plat.c('[>]', 'cyan')} ComfyScript (editable, [default] extras) ...")
        if plat.run([py, "-m", "pip", "install", "-e", f"{cs}[default]"]) == 0:
            record("ComfyScript", "installed (editable)")
        else:
            record("ComfyScript", "failed")
            failed.append("ComfyScript")

    print("\n  Summary:")
    for status, count in sorted(results.items()):
        print(f"    {status}: {count}")
    if failed:
        plat.warn("failed packs (re-run pip manually):")
        for f in failed:
            print(f"    {plat.c('-', 'yellow')} {f}")
        return 1
    return 0

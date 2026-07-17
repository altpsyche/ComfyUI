"""Pack registry + the build / models / validate commands. STDLIB ONLY.

Adding a pack = drop a `pack.toml` + a build() adapter, then register it here. See the
"Adding a new pack" section in tools/il_graphs/ARCHITECTURE.md.
"""
from __future__ import annotations

import importlib
import sys

from ..core import config
from ..core import nodes
from ..core import platform as plat
from .base import Context

# name -> module exposing PACK (a Pack subclass)
REGISTRY = {
    "il_graphs": "devtools.packs.il_graphs",
}


def _get(name):
    if name not in REGISTRY:
        plat.err(f"unknown pack {name!r}; registered: {', '.join(sorted(REGISTRY))}")
        return None
    mod = importlib.import_module(REGISTRY[name])
    return mod.PACK()


def _check_nodes(meta) -> None:
    if not meta.custom_nodes:
        return
    missing, mismatched = nodes.check(meta.custom_nodes)
    if missing:
        plat.warn(f"custom_nodes missing/uninitialized: {', '.join(missing)} "
                  "(run: git submodule update --init)")
    if mismatched:
        plat.warn(f"custom_nodes pinned to a specific SHA that differs: {', '.join(mismatched)}")
    if not missing and not mismatched:
        plat.ok(f"{len(meta.custom_nodes)} required custom_nodes present")


def build(name) -> int:
    if name == "all":
        rc = 0
        for n in REGISTRY:
            rc |= build(n)
        return rc
    pack = _get(name)
    if pack is None:
        return 1
    plat.heading(f"build: {name}")
    _check_nodes(pack.meta)
    return pack.build(Context())


def models_install(name, *, variants=None, with_optional=False) -> int:
    pack = _get(name)
    if pack is None:
        return 1
    if not pack.meta.models_manifest or not pack.meta.models_manifest.exists():
        plat.err(f"pack {name!r} declares no models_manifest")
        return 1
    from ..core import download
    plat.heading(f"models: {name}")
    return download.install(pack.meta.models_manifest, config.MODELS,
                            variants=variants, with_optional=with_optional)


def validate(workflow) -> int:
    tools = str(config.TOOLS)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from validate_workflow import validate as _validate
    return _validate(workflow, require_models=False, require_wildcards=True)

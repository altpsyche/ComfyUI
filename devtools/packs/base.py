"""The Pack contract: declarative metadata (pack.toml) + one code method, build(ctx).

A "pack" is a self-contained tool area — anime characters (il_graphs) today; realistic characters,
3D, texture generation later. Everything that differs per pack is data in pack.toml; the only code
a pack must supply is how to generate its ComfyUI workflows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core import config


@dataclass
class PackMeta:
    name: str
    kind: str = "image-char"           # image-char | realistic-char | 3d | texture
    output_subdir: str = ""            # "" = flat workflows dir with a name prefix
    schema_version: int = 1            # stamped into generated JSON; bump on generator changes
    models_manifest: Path | None = None
    custom_nodes: list = field(default_factory=list)   # [{name, sha}] — checked by verify/build
    train: dict | None = None          # optional [train] profile; omit -> pack has no `dev train`
    dir: Path | None = None            # directory containing pack.toml

    @classmethod
    def load(cls, toml_path) -> "PackMeta":
        toml_path = Path(toml_path)
        doc = config.load_toml(toml_path)
        manifest = doc.get("models_manifest")
        return cls(
            name=doc["name"],
            kind=doc.get("kind", "image-char"),
            output_subdir=doc.get("output_subdir", ""),
            schema_version=int(doc.get("schema_version", 1)),
            models_manifest=(toml_path.parent / manifest) if manifest else None,
            custom_nodes=doc.get("custom_nodes", []),
            train=doc.get("train"),
            dir=toml_path.parent,
        )


@dataclass
class Context:
    """Passed to build(). Packs read these instead of recomputing repo-relative paths."""
    root: Path = config.ROOT
    models_dir: Path = config.MODELS
    output_dir: Path = config.OUTPUT


class Pack:
    """Base class. A concrete pack sets `toml` (path to its pack.toml) and implements build()."""
    toml: Path

    def __init__(self):
        self.meta = PackMeta.load(self.toml)

    def build(self, ctx: Context) -> int:
        raise NotImplementedError

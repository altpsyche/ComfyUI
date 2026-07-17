"""Generic, variant-aware, idempotent model downloader (Hugging Face CLI). STDLIB ONLY.

Generalizes the old install_qwen_edit.ps1: a pack declares its models in a `models.toml`
manifest; this fetches each entry that isn't already on disk. Shared entries (a VAE, a text
encoder) reuse the HF cache (HF_HOME), so two packs listing the same file don't re-download it.

Manifest shape (see tools/il_graphs/models.toml):

    [[models]]
    repo = "org/repo"          # HF repo id
    src  = "path/in/repo.safetensors"
    dest = "vae"               # subdir under models/
    name = "override.safetensors"   # optional; defaults to basename(src)
    optional = true            # optional; skipped unless --with-optional

    [[variants]]              # a group where the choice changes the filename
    group   = "quant"
    repo    = "org/repo-gguf"
    dest    = "unet"
    pattern = "model-{variant}.gguf"     # {variant} substituted with the chosen value
    src_dir = ""              # optional path prefix inside the repo
    choices = ["Q4_K_M", "Q5_K_M"]
    default = "Q5_K_M"
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from . import config
from . import platform as plat


def _hf():
    """Resolve the HF CLI: prefer the main venv's copy, then PATH. Returns a path str or None."""
    from . import venv

    for handle_name in (("main", "hf"), ("main", "huggingface-cli")):
        cand = venv.bin(*handle_name)
        if cand.exists():
            return str(cand)
    for name in ("hf", "huggingface-cli"):
        p = plat.have(name)
        if p:
            return p
    return None


def _jobs(doc, chosen, with_optional):
    """Flatten a manifest doc into (repo, src, dest_subdir, dest_name) tuples."""
    jobs = []
    for m in doc.get("models", []):
        if m.get("optional") and not with_optional:
            continue
        jobs.append((m["repo"], m["src"], m["dest"], m.get("name") or Path(m["src"]).name))
    for v in doc.get("variants", []):
        group = v["group"]
        choice = (chosen or {}).get(group) or v.get("default")
        if choice not in v.get("choices", []):
            raise SystemExit(
                f"unknown --variant {group}={choice!r}; choices: {', '.join(v.get('choices', []))}"
            )
        fname = v["pattern"].format(variant=choice)
        src = f"{v['src_dir'].rstrip('/')}/{fname}" if v.get("src_dir") else fname
        jobs.append((v["repo"], src, v["dest"], fname))
    return jobs


def _get(hf, repo, src, dest: Path) -> bool:
    """Download one file into `dest` (idempotent). Stages to a temp dir then moves, flattening any
    nested repo path. Returns True on success (or already-present)."""
    if dest.exists():
        plat.info(f"{dest.relative_to(config.MODELS.parent)} already present")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="dvdl_"))
    plat.step(f"{repo} :: {src}")
    rc = plat.run([hf, "download", repo, src, "--local-dir", str(stage)])
    if rc != 0:
        shutil.rmtree(stage, ignore_errors=True)
        plat.err(f"download failed: {repo}/{src}")
        return False
    shutil.move(str(stage / src), str(dest))
    shutil.rmtree(stage, ignore_errors=True)
    plat.ok(f"{dest.relative_to(config.MODELS.parent)}")
    return True


def install(manifest_path, models_dir=None, *, variants=None, with_optional=False) -> int:
    """Fetch every manifest entry not already on disk. Returns 0 if all succeeded."""
    models_dir = Path(models_dir or config.MODELS)
    doc = config.load_toml(manifest_path)
    hf = _hf()
    if not hf:
        plat.err("Hugging Face CLI not found (need 'hf' or 'huggingface-cli'; install into the venv)")
        return 1
    failed = 0
    for repo, src, dest_sub, name in _jobs(doc, variants, with_optional):
        if not _get(hf, repo, src, models_dir / dest_sub / name):
            failed += 1
    return 1 if failed else 0

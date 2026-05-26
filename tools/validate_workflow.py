"""Pre-flight validator for a ComfyUI workflow JSON (any version).

Reads MainGraph*.json (or any workflow), checks:
  - Model files referenced exist on disk (checkpoints, VAE, LoRAs, embeddings,
    ControlNets, SAM, Ultralytics detectors, upscale models, IPAdapter)
  - Hard constraints (CLIPSetLastLayer == -2, CFGGuider >= 5)
  - Wildcard files exist (Impact-Pack `__name__` tokens)

Operates on the workflow JSON directly — does NOT require workflows-src/.
Author your workflow in ComfyUI UI, save, run this to catch issues before launch.

Usage:
    python tools/validate_workflow.py user/default/workflows/MainGraphv8.json
    python tools/validate_workflow.py user/default/workflows/MainGraphv9.json --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
CUSTOM_NODES = REPO_ROOT / "custom_nodes"
WILDCARDS_DIRS = [
    CUSTOM_NODES / "ComfyUI-Impact-Pack" / "wildcards",
    CUSTOM_NODES / "ComfyUI-Inspire-Pack" / "wildcards",
]


# Map node type → (widget_index, models_subdir). None means dynamic logic.
SIMPLE_MODEL_LOADERS = {
    "CheckpointLoaderSimple":     (0, "checkpoints"),
    "VAELoader":                  (0, "vae"),
    "UpscaleModelLoader":         (0, "upscale_models"),
    "ControlNetLoader":           (0, "controlnet"),
    "SAMLoader":                  (0, "sams"),
    "UltralyticsDetectorProvider":(0, "ultralytics"),
    "LoraLoader":                 (0, "loras"),
    "LoraLoaderModelOnly":        (0, "loras"),
    "IPAdapterUnifiedLoader":     None,   # widget = preset name, not a file
    "DiffControlNetLoader":       (0, "controlnet"),
    "DualCLIPLoader":             None,   # multiple clip files, skip
    "CLIPVisionLoader":           (0, "clip_vision"),
}


def find_models(g: dict) -> list[tuple[str, str, int]]:
    """Return list of (node_id, expected_relative_path, node_index_in_widgets).

    Walks every node, applies known mappings. Skips dynamic loaders.
    """
    refs: list[tuple[str, str, int]] = []
    for n in g.get("nodes", []):
        nt = n.get("type")
        if nt in SIMPLE_MODEL_LOADERS:
            mapping = SIMPLE_MODEL_LOADERS[nt]
            if mapping is None:
                continue
            widget_idx, subdir = mapping
            widgets = n.get("widgets_values") or []
            if widget_idx >= len(widgets):
                continue
            name = widgets[widget_idx]
            if not isinstance(name, str) or not name:
                continue
            refs.append((str(n["id"]), f"{subdir}/{name}", widget_idx))

        # Power Lora Loader (rgthree): widgets is list of dicts
        if nt == "Power Lora Loader (rgthree)":
            for w in n.get("widgets_values") or []:
                if isinstance(w, dict) and w.get("on") and w.get("lora"):
                    refs.append((str(n["id"]), f"loras/{w['lora']}", -1))

    return refs


def find_embedding_tokens(g: dict) -> list[tuple[str, str]]:
    """Return list of (node_id, embedding_name). Scans CLIPTextEncode for `embedding:NAME`."""
    out = []
    for n in g.get("nodes", []):
        if n.get("type") != "CLIPTextEncode":
            continue
        widgets = n.get("widgets_values") or []
        if not widgets or not isinstance(widgets[0], str):
            continue
        for m in re.finditer(r"embedding:([\w.\-]+)", widgets[0]):
            out.append((str(n["id"]), m.group(1)))
    return out


def find_wildcard_tokens(g: dict) -> list[tuple[str, str]]:
    """Return (node_id, token_name) for `__name__` references in FaceDetailer/SEGSDetailer wildcards."""
    out = []
    for n in g.get("nodes", []):
        nt = n.get("type")
        if nt not in ("FaceDetailer", "SEGSDetailer"):
            continue
        widgets = n.get("widgets_values") or []
        # FaceDetailer wildcard at index 23, SEGSDetailer doesn't have one
        # (but some forks might — scan all string widgets)
        for w in widgets:
            if isinstance(w, str):
                for m in re.finditer(r"__([\w-]+)__", w):
                    out.append((str(n["id"]), m.group(1)))
    return out


def check_hard_constraints(g: dict) -> list[str]:
    """Return list of constraint violation messages."""
    errs = []
    for n in g.get("nodes", []):
        if n.get("mode") in (2, 4):
            continue  # muted / bypassed — skip
        nt = n.get("type")
        widgets = n.get("widgets_values") or []
        if nt == "CLIPSetLastLayer" and widgets:
            v = widgets[0]
            if v != -2:
                errs.append(f"node {n['id']} CLIPSetLastLayer = {v}, expected -2 "
                            "(amanatsuIllustrious hard requirement)")
        if nt == "CFGGuider" and widgets:
            v = widgets[0]
            try:
                vf = float(v)
                if vf < 5.0:
                    errs.append(f"node {n['id']} CFGGuider = {v}, expected >= 5.0 "
                                "(amanatsuIllustrious composition floor)")
            except (TypeError, ValueError):
                pass
    return errs


def find_lora_widget_for_loader(g: dict) -> list[tuple[str, str]]:
    """Stub for completeness — already covered by SIMPLE_MODEL_LOADERS and rgthree."""
    return []


def validate(path: Path, *, strict: bool = False) -> int:
    g = json.loads(path.read_text(encoding="utf-8"))

    print(f"workflow: {path}")
    print(f"  {len(g.get('nodes', []))} nodes, {len(g.get('links', []))} links")
    print()

    missing_files: list[tuple[str, str]] = []
    missing_emb: list[tuple[str, str]] = []
    missing_wc: list[tuple[str, str]] = []
    constraint_errs: list[str] = []

    # 1. Model files
    for node_id, rel_path, _ in find_models(g):
        p = MODELS / rel_path
        if not p.exists():
            missing_files.append((node_id, rel_path))

    # 2. Embedding tokens
    for node_id, emb_name in find_embedding_tokens(g):
        candidates = [MODELS / "embeddings" / f"{emb_name}.safetensors",
                      MODELS / "embeddings" / f"{emb_name}.pt",
                      MODELS / "embeddings" / f"{emb_name}.bin"]
        if not any(c.exists() for c in candidates):
            missing_emb.append((node_id, emb_name))

    # 3. Wildcards
    for node_id, tok in find_wildcard_tokens(g):
        found = False
        for wd in WILDCARDS_DIRS:
            if (wd / f"{tok}.txt").exists():
                found = True
                break
        if not found:
            missing_wc.append((node_id, tok))

    # 4. Hard constraints
    constraint_errs = check_hard_constraints(g)

    total = len(missing_files) + len(missing_emb) + len(missing_wc) + len(constraint_errs)
    if total == 0:
        print("[+] all checks passed")
        return 0

    if missing_files:
        print(f"missing model files ({len(missing_files)}):")
        for nid, rel in missing_files:
            print(f"  - node {nid}: models/{rel}")
        print()

    if missing_emb:
        print(f"missing embeddings ({len(missing_emb)}):")
        for nid, name in missing_emb:
            print(f"  - node {nid}: models/embeddings/{name}.(safetensors|pt|bin)")
        print()

    if missing_wc:
        print(f"missing wildcard files ({len(missing_wc)}):")
        for nid, tok in missing_wc:
            print(f"  - node {nid}: __{tok}__ "
                  f"-> custom_nodes/ComfyUI-Impact-Pack/wildcards/{tok}.txt")
        print()

    if constraint_errs:
        print(f"hard-constraint violations ({len(constraint_errs)}):")
        for e in constraint_errs:
            print(f"  - {e}")
        print()

    return 1 if (strict or constraint_errs or missing_files) else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("workflow", help="path to a ComfyUI workflow JSON")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any issue (default: only on missing files / constraints)")
    args = p.parse_args()

    path = Path(args.workflow)
    if not path.exists():
        print(f"[x] not found: {path}", file=sys.stderr)
        return 1

    return validate(path, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())

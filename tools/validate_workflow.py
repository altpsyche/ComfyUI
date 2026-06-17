"""Pre-flight validator for a ComfyUI workflow JSON (any version).

Generic checks (run on every workflow):
  - Model files referenced exist on disk (checkpoints, VAE, LoRAs, embeddings,
    ControlNets, SAM, Ultralytics detectors, upscale models, IPAdapter, CLIP vision)
  - Wildcard files exist (Impact-Pack `__name__` tokens)
  - Power Lora Loader (rgthree) enabled-LoRA file existence

Optional per-workflow rules (auto-loaded from sidecar):
  Place `<workflow>.rules.toml` next to the .json. Recognized rules:
    clip_skip     = -2          # require all active CLIPSetLastLayer at -2
    min_cfg       = 5.0         # require all active CFGGuider >= 5.0
    max_cfg       = 12.0        # require all active CFGGuider <= 12.0
    require_nodes = ["X", ...]  # warn if any of these node types missing
    forbid_nodes  = ["Y", ...]  # warn if any of these node types present

Usage:
    python tools/validate_workflow.py user/default/workflows/MainGraphv8.json
    python tools/validate_workflow.py user/default/workflows/MainGraphv9.json --strict
    python tools/validate_workflow.py v8.json --rules path/to/rules.toml
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
    "UnetLoaderGGUF":             (0, "unet"),          # ComfyUI-GGUF; widget0 = gguf in models/unet
    "CLIPLoader":                 (0, "text_encoders"), # widget0 = text encoder file
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
    """Return (node_id, token_name) for `__name__` references in wildcard-bearing nodes.

    Covers FaceDetailer/SEGSDetailer (their wildcard widget) AND the Impact-Pack wildcard
    processor/encoder used by the IL_DatasetEdit graphs — otherwise a missing __pose__/__angle__
    file passes pre-flight and only fails at generation time. Scans all string widgets (the
    wildcard text + the populated text both hold the tokens) and dedups per node.
    """
    out = []
    WILDCARD_NODES = ("FaceDetailer", "SEGSDetailer",
                      "ImpactWildcardProcessor", "ImpactWildcardEncode")
    for n in g.get("nodes", []):
        if n.get("type") not in WILDCARD_NODES:
            continue
        seen = set()
        for w in n.get("widgets_values") or []:
            if isinstance(w, str):
                for m in re.finditer(r"__([\w-]+)__", w):
                    if m.group(1) not in seen:
                        seen.add(m.group(1))
                        out.append((str(n["id"]), m.group(1)))
    return out


def load_rules(workflow_path: Path, override: Path | None = None) -> dict:
    """Auto-load sidecar `<workflow>.rules.toml` if present, or use override path."""
    if override is not None:
        candidate = override
    else:
        candidate = workflow_path.with_suffix(".rules.toml")
    if not candidate.exists():
        return {}
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        import tomli as tomllib  # py3.10
    try:
        return tomllib.loads(candidate.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[!] failed to parse {candidate}: {e}", file=sys.stderr)
        return {}


def check_rules(g: dict, rules: dict) -> list[str]:
    """Apply optional per-workflow rules. Returns violations."""
    errs: list[str] = []
    if not rules:
        return errs

    active_nodes = [n for n in g.get("nodes", []) if n.get("mode") not in (2, 4)]
    active_types = {n.get("type") for n in active_nodes}

    clip_skip_target = rules.get("clip_skip")
    if clip_skip_target is not None:
        for n in active_nodes:
            if n.get("type") == "CLIPSetLastLayer":
                widgets = n.get("widgets_values") or []
                if widgets and widgets[0] != clip_skip_target:
                    errs.append(f"node {n['id']} CLIPSetLastLayer = {widgets[0]}, "
                                f"rule requires {clip_skip_target}")

    # cfg sits at a different widget index per node type. CFGGuider alone is not enough — the IL
    # family drives cfg through KSampler / UltimateSDUpscale / detailers, so check those too or the
    # rule is vacuous. Indices verified against the live node widget order.
    CFG_WIDGET_IDX = {
        "CFGGuider": 0,          # [cfg]
        "KSampler": 3,           # [seed, control, steps, CFG, sampler, scheduler, denoise]
        "KSamplerAdvanced": 4,   # [add_noise, seed, control, steps, CFG, sampler, scheduler, ...]
        "UltimateSDUpscale": 4,  # [upscale_by, seed, control, steps, CFG, sampler, scheduler, denoise, ...]
        "FaceDetailer": 6,       # [guide, guide_for, max, seed, control, steps, CFG, sampler, ...]
        "SEGSDetailer": 6,       # [guide, guide_for, max, seed, control, steps, CFG, sampler, ...]
    }
    min_cfg = rules.get("min_cfg")
    max_cfg = rules.get("max_cfg")
    if min_cfg is not None or max_cfg is not None:
        for n in active_nodes:
            idx = CFG_WIDGET_IDX.get(n.get("type"))
            if idx is None:
                continue
            widgets = n.get("widgets_values") or []
            if idx >= len(widgets):
                continue
            try:
                v = float(widgets[idx])
            except (TypeError, ValueError):
                continue
            if min_cfg is not None and v < min_cfg:
                errs.append(f"node {n['id']} {n['type']} cfg = {v}, rule requires >= {min_cfg}")
            if max_cfg is not None and v > max_cfg:
                errs.append(f"node {n['id']} {n['type']} cfg = {v}, rule requires <= {max_cfg}")

    require_nodes = rules.get("require_nodes") or []
    for needed in require_nodes:
        if needed not in active_types:
            errs.append(f"required node type '{needed}' missing (or all instances bypassed)")

    forbid_nodes = rules.get("forbid_nodes") or []
    for banned in forbid_nodes:
        if banned in active_types:
            errs.append(f"forbidden node type '{banned}' present and active")

    return errs


def validate(path: Path, *, strict: bool = False, rules_path: Path | None = None,
             require_models: bool = True, require_wildcards: bool = False) -> int:
    """Validate one workflow. Returns 0 (ok) / 1 (fail).

    `require_models` (default True) → missing model files count as failures, matching CLI behaviour.
    `require_wildcards` (default False) → missing `__token__` files count as failures. The build-time
    guardrail calls validate(require_models=False, require_wildcards=True) so a fresh checkout missing
    large model downloads still builds, while rule/wildcard regressions hard-fail.
    """
    g = json.loads(path.read_text(encoding="utf-8"))
    rules = load_rules(path, rules_path)

    print(f"workflow: {path}")
    print(f"  {len(g.get('nodes', []))} nodes, {len(g.get('links', []))} links")
    if rules:
        rules_src = rules_path or path.with_suffix(".rules.toml")
        print(f"  rules:    {rules_src} ({len(rules)} rule(s))")
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

    # 4. Optional per-workflow rules
    constraint_errs = check_rules(g, rules)

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
        print(f"rule violations ({len(constraint_errs)}):")
        for e in constraint_errs:
            print(f"  - {e}")
        print()

    hard = bool(constraint_errs)
    if require_models:
        hard = hard or bool(missing_files)
    if require_wildcards:
        hard = hard or bool(missing_wc)
    return 1 if (strict or hard) else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("workflow", help="path to a ComfyUI workflow JSON")
    p.add_argument("--rules",
                   help="path to a rules.toml (default: auto-load <workflow>.rules.toml)")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any issue (default: only on missing files / rules)")
    args = p.parse_args()

    path = Path(args.workflow)
    if not path.exists():
        print(f"[x] not found: {path}", file=sys.stderr)
        return 1

    rules_path = Path(args.rules) if args.rules else None
    return validate(path, strict=args.strict, rules_path=rules_path)


if __name__ == "__main__":
    sys.exit(main())

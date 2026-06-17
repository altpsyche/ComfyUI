"""Build all IL_* workflows: write each {name}.json + {name}.rules.toml + {name}.md.

Run:  python tools/build_il_graphs.py   (or:  python -m il_graphs.build)
"""
from __future__ import annotations
import json
import sys
from .config import OUT, ROOT, CHARACTERS, SEED
from .docs import md
from .graphs import (build_base, build_refine, build_guided, build_studio,
                     build_max, build_ipadapter, build_pose, build_lcm,
                     build_xyplot, build_dataset_edit)


def main():
    graphs = {"IL_1_Base": build_base(), "IL_2_Refine": build_refine(), "IL_3_Guided": build_guided(),
              "IL_4_Studio": build_studio(), "IL_5_Max": build_max(),
              "IL_IPAdapter": build_ipadapter(), "IL_Pose": build_pose(), "IL_LCM": build_lcm(),
              "IL_XYPlot": build_xyplot()}
    base_req = ["CheckpointLoaderSimple", "CLIPSetLastLayer", "KSampler", "VAEDecode", "SaveImage"]
    det_req = base_req + ["UltimateSDUpscale", "FaceDetailer"]
    req = {"IL_1_Base": base_req,
           "IL_2_Refine": base_req + ["UltimateSDUpscale"],
           "IL_3_Guided": det_req,
           "IL_4_Studio": det_req + ["ACN_AdvancedControlNetApply_v2", "SEGSDetailer"],
           "IL_5_Max": det_req + ["ACN_AdvancedControlNetApply_v2", "SEGSDetailer",
                                  "InpaintCropImproved", "InpaintStitchImproved"],
           "IL_IPAdapter": det_req + ["IPAdapterAdvanced"],
           "IL_Pose": det_req + ["ControlNetApplyAdvanced", "OpenposeEditorNode"],
           "IL_LCM": base_req,   # cfg 1.5 ok: min_cfg rule omitted for LCM (low CFG by design)
           "IL_XYPlot": ["Efficient Loader", "KSampler (Efficient)", "XY Plot",
                         "XY Input: LoRA Plot", "SaveImage"]}
    # Qwen-Image-Edit graph: no checkpoint/clip-skip (clip_skip & min_cfg rules are vacuous here --
    # no CLIPSetLastLayer, no CFGGuider; KSampler cfg 1.0 is intended for Lightning).
    edit_req = ["CheckpointLoaderSimple", "UnetLoaderGGUF", "CLIPLoader", "TextEncodeQwenImageEditPlus",
                "FluxKontextMultiReferenceLatentMethod", "KSampler", "VAEDecode", "SaveImage"]

    # roster.json (name/trigger/id/outfit/prune) is the trainer's source of truth. Each character gets
    # exactly one dataset workflow: the self-contained Qwen-Image-Edit graph IL_DatasetEdit_<name>.
    # `like: "<other>"` inherits that entry's id + hero_seed so a same-face/different-outfit variant
    # (e.g. aria_gala like aria) needs only its own outfit -- same hero face, separate locked LoRA.
    # roster carries `outfit` so train_lora auto-bakes it into the trigger (no manual prune chasing).
    roster = []
    for cname, raw in CHARACTERS.items():
        spec = dict(raw)
        if "like" in spec:
            parent = CHARACTERS[spec["like"]]
            spec.setdefault("id", parent["id"])
            spec.setdefault("hero_seed", parent.get("hero_seed", SEED))
        hero_seed = spec.get("hero_seed", SEED)
        outfit = spec.get("outfit", "")
        roster.append({"name": cname, "trigger": spec.get("trigger") or f"{cname}char",
                       "id": spec["id"], "outfit": outfit, "prune": spec.get("prune", "")})
        egname = f"IL_DatasetEdit_{cname}"
        graphs[egname] = build_dataset_edit(cname, spec["id"], outfit, hero_seed)
        req[egname] = edit_req
    (ROOT / "tools/lora_train/roster.json").write_text(json.dumps(roster, indent=2), encoding="utf-8")

    for name, g in graphs.items():
        (OUT / f"{name}.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
        rlist = "".join(f'    "{n}",\n' for n in req[name])
        # The CFG floor is now actually enforced (validate_workflow checks KSampler/USDU/detailer cfg,
        # not just CFGGuider). LCM and the Qwen-Edit dataset graphs intentionally run a low CFG
        # (LCM / Lightning), so they get NO min_cfg rule; everything else keeps the >= 5 hard floor.
        low_cfg = name == "IL_LCM" or name.startswith("IL_DatasetEdit")
        no_clipnode = name == "IL_XYPlot"   # uses Efficient Loader's clip_skip widget, not a CLIPSetLastLayer
        cfg_rules = "" if low_cfg else "min_cfg = 5.0\nmax_cfg = 12.0\n"
        cs_rule = "" if no_clipnode else "clip_skip = -2\n"
        cs_comment = ("# CLIP skip is set in the Efficient Loader widget (-2), no CLIPSetLastLayer to check.\n"
                      if no_clipnode else
                      "# Hard requirement for amanatsu/oneObsession Illustrious: CLIP skip -2.\n")
        cfg_comment = ("# Low CFG by design (LCM / Lightning) — no min_cfg rule here.\n" if low_cfg
                       else "# Hard floor: CFG >= 5 on every sampler (KSampler / USDU / detailers).\n")
        (OUT / f"{name}.rules.toml").write_text(
            f"# Validator rules for {name}.json — auto-loaded by tools/validate_workflow.py\n"
            f"{cs_comment}{cfg_comment}"
            f"{cs_rule}{cfg_rules}require_nodes = [\n{rlist}]\n",
            encoding="utf-8")
        (OUT / f"{name}.md").write_text(md(name, g), encoding="utf-8")
        print(f"wrote {name}.json: {len(g['nodes'])} nodes, {len(g['links'])} links  (+ rules.toml + md)")

    # Prune stale dataset-family workflows (character removed/renamed, plus any legacy IL_Dataset_<name>
    # from the retired hero+IPAdapter route) so the ComfyUI list doesn't keep orphans. The IL_Dataset*
    # glob keeps the live IL_DatasetEdit_* (they're in `graphs`) and drops everything else it matches.
    removed = []
    for jf in OUT.glob("IL_Dataset*.json"):
        if jf.stem not in graphs:
            for fp in (jf, OUT / f"{jf.stem}.rules.toml", OUT / f"{jf.stem}.md"):
                if fp.exists():
                    fp.unlink()
            removed.append(jf.stem)
    if removed:
        print(f"removed stale: {', '.join(sorted(removed))}")

    # Prune stale dataset .toml caches (character removed/renamed) so train_lora can't reuse a config
    # for a character that no longer exists. The roster (CHARACTERS) is the source of truth.
    cache_dir = ROOT / "tools/lora_train/.cache"
    if cache_dir.exists():
        valid = set(CHARACTERS)
        stale_cache = [tf for tf in cache_dir.glob("*.toml") if tf.stem not in valid]
        for tf in stale_cache:
            tf.unlink()
        if stale_cache:
            print(f"removed stale .cache: {', '.join(sorted(t.stem for t in stale_cache))}")

    # --- build-time validation guardrail ---
    # Validate every emitted graph against its own rules + wildcards. Missing MODEL files are a warning
    # (a fresh checkout won't have the multi-GB downloads); rule and wildcard violations hard-fail the
    # build so a broken graph can't slip through silently. Skip with --no-validate.
    if "--no-validate" not in sys.argv:
        import io
        import contextlib
        try:
            from validate_workflow import validate
        except ImportError:                       # support `python -m il_graphs.build`
            sys.path.insert(0, str(ROOT / "tools"))
            from validate_workflow import validate
        print()
        failed = []
        for name in graphs:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = validate(OUT / f"{name}.json", require_models=False, require_wildcards=True)
            if rc == 0:
                print(f"[validate] {name} ... OK")
            else:
                print(f"[validate] {name} ... FAIL")
                print(buf.getvalue().rstrip())
                failed.append(name)
        if failed:
            print(f"\n[x] validation failed: {', '.join(failed)}")
            sys.exit(1)
        print(f"[+] {len(graphs)} graphs built + validated")


if __name__ == "__main__":
    main()

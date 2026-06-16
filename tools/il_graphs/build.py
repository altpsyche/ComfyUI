"""Build all IL_* workflows: write each {name}.json + {name}.rules.toml + {name}.md.

Run:  python tools/build_il_graphs.py   (or:  python -m il_graphs.build)
"""
from __future__ import annotations
import json
from .config import OUT, ROOT, CHARACTERS
from .docs import md
from .graphs import (build_base, build_refine, build_guided, build_studio,
                     build_max, build_ipadapter, build_pose, build_lcm,
                     build_dataset_edit)


def main():
    graphs = {"IL_1_Base": build_base(), "IL_2_Refine": build_refine(), "IL_3_Guided": build_guided(),
              "IL_4_Studio": build_studio(), "IL_5_Max": build_max(),
              "IL_IPAdapter": build_ipadapter(), "IL_Pose": build_pose(), "IL_LCM": build_lcm()}
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
           "IL_LCM": base_req}   # cfg 1.5 ok: min_cfg rule only checks CFGGuider, LCM uses KSampler
    # Qwen-Image-Edit graph: no checkpoint/clip-skip (clip_skip & min_cfg rules are vacuous here --
    # no CLIPSetLastLayer, no CFGGuider; KSampler cfg 1.0 is intended for Lightning).
    edit_req = ["CheckpointLoaderSimple", "UnetLoaderGGUF", "CLIPLoader", "TextEncodeQwenImageEditPlus",
                "FluxKontextMultiReferenceLatentMethod", "KSampler", "VAEDecode", "SaveImage"]

    # roster.json (name/trigger/prune) is the trainer's source of truth. Each character gets exactly one
    # dataset workflow: the self-contained Qwen-Image-Edit graph IL_DatasetEdit_<name>.
    roster = []
    for cname, spec in CHARACTERS.items():
        roster.append({"name": cname, "trigger": spec.get("trigger") or f"{cname}char",
                       "prune": spec.get("prune", "")})
        egname = f"IL_DatasetEdit_{cname}"
        graphs[egname] = build_dataset_edit(cname, spec["id"], spec.get("outfit", ""))
        req[egname] = edit_req
    (ROOT / "tools/lora_train/roster.json").write_text(json.dumps(roster, indent=2), encoding="utf-8")

    for name, g in graphs.items():
        (OUT / f"{name}.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
        rlist = "".join(f'    "{n}",\n' for n in req[name])
        (OUT / f"{name}.rules.toml").write_text(
            f"# Validator rules for {name}.json — auto-loaded by tools/validate_workflow.py\n"
            f"# Hard requirement for amanatsu/oneObsession Illustrious: CLIP skip -2.\n"
            f"clip_skip = -2\nmin_cfg = 5.0\nmax_cfg = 12.0\nrequire_nodes = [\n{rlist}]\n",
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


if __name__ == "__main__":
    main()

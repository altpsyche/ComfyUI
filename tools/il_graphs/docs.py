from __future__ import annotations
from .config import CKPT, POS, NEG, HAND_POS, FACE_POS, CHAR

PROMPT_INFO = {
  "Positive": ("base generation — and is reused by the upscaler (USDU) and the face detailer; "
               "there is no separate upscaler prompt", POS),
  "Negative": ("base generation, the upscaler, and every detailer", NEG),
  "Hand positive": ("the hand detailer only (re-renders hands)", HAND_POS),
  "Face positive": ("the native-1024 face-inpaint pass only", FACE_POS),
  "Hero portrait prompt": ("the fixed-seed hero portrait that becomes the IPAdapter face source — "
     "edit CHAR in config to your character's weighted identity tags",
     "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles, cream knit sweater, "
     "blue jeans, upper body, plain grey background, character portrait"),
  "Face detail (neutral identity)": ("the face detailer — neutral (no pose/gaze tags) so the "
     "re-rolled face crop doesn't fight the body's pose", CHAR),
}


DOCS = {
  "IL_1_Base": ("Plain txt2img — the reference image. No upscale, no detailers.",
    "Fast drafts, prompt testing, and the baseline every other tier builds on.",
    ["Edit the Positive / Negative prompt nodes.",
     "Seed is fixed at 1234567890 (change it in the Seed node to re-roll).", "Queue."],
    ["KSampler: steps 30 / cfg 5 / euler_ancestral / normal.",
     "Empty Latent 832x1216 (SDXL portrait native)."]),
  "IL_2_Refine": ("IL_1 base + 1.5x UltimateSDUpscale + face detailer. Same base image as IL_1.",
    "Normal single images that need resolution and a clean face.",
    ["Edit prompts. Nothing to load.", "Queue."],
    ["USDU: upscale_by 1.5 / denoise 0.20 / Linear / Half Tile / 4x-AnimeSharp.",
     "FaceDetailer denoise 0.3 (face_yolov9c + sam2)."]),
  "IL_3_Guided": ("IL_2 + hand detailer. Face and hand cleanup on the upscaled image.",
    "Subjects where hands are visible and matter.",
    ["Edit prompts. Nothing to load.", "Queue."],
    ["Hand detailer denoise 0.3; detector bbox thresholds.",
     "Hand positive is a separate node so it doesn't fight the scene prompt."]),
  "IL_4_Studio": ("IL_3 + depth+lineart ControlNet at upscale (structure lock) + background detailer "
    "+ aesthetic score + metadata save.",
    "Final / portfolio renders. Locks structure during the upscale pass.",
    ["Edit prompts. Nothing to load.", "Queue. Aesthetic score shows in the node after gen."],
    ["CN apply strengths: depth 0.35 / lineart 0.25, soft weights 0.825.",
     "Background detailer runs on the inverted person mask."]),
  "IL_5_Max": ("IL_4 + native-1024 face-inpaint chain + sharpen. Highest single-image quality. "
    "Zero bypassed nodes.",
    "Best quality for a single image.",
    ["Edit prompts. Nothing to load.", "Queue."],
    ["Face inpaint KSampler denoise 0.45 on the cropped 1024 face.",
     "Lucy sharpen at the end."]),
  "IL_IPAdapter": ("Production pipeline (base + upscale + detailers) with IPAdapter plus-face ACTIVE "
    "for style/face transfer from a reference image.",
    "When you want to carry a specific face or style from a reference.",
    ["LOAD a face/style image in the 'IPAdapter ref >> LOAD' node.",
     "Edit prompts. Queue."],
    ["IPAdapter weight 0.7 (raise for stronger identity, lower if it overrides the prompt)."]),
  "IL_Pose": ("Production pipeline with OpenPose ControlNet ACTIVE on the base generation.",
    "When you want to control the subject's pose from a reference image.",
    ["LOAD a pose image in the 'Pose ref >> LOAD' node (DWPose extracts the skeleton).",
     "Edit prompts. Queue."],
    ["Pose apply strength 0.7 / end 0.8. Edit the skeleton in the Pose edit node if needed."]),
  "IL_LCM": ("Fast preview — lcm-lora + LCM sampler (8 steps, cfg 1.5). Base only, no heavy post.",
    "Quick composition / prompt checks (~4x faster, lower quality). Switch to a full tier for finals.",
    ["Edit prompts. Queue."],
    ["KSampler: lcm / sgm_uniform / 8 steps / cfg 1.5. lcm-lora ON in the loader."]),
  "IL_Dataset": ("Synthetic training-data generator for ONE character. A fixed-seed hero portrait "
    "feeds an IPAdapter PLUS-FACE that pins that face onto every render; an Impact wildcard prompt "
    "varies pose/angle/framing/expression and the Gen Seed re-rolls. No external image loads.",
    "Step 1 of training a character-consistency LoRA: generate a varied, on-model image set to caption and train on.",
    ["Edit CHAR in tools/il_graphs/config.py to your character's weighted identity tags, then regenerate.",
     "Run with Hero Seed FIXED; pick a hero portrait you like — it becomes the locked face.",
     "Reroll the Gen Seed repeatedly (batch of 4) to fill output/dataset/ with ~60 varied shots.",
     "Curate the on-model ~30, caption (WD14 + a unique trigger token), then train the LoRA.",
     "Load the trained LoRA in any IL workflow's LoRA bank (toggle on + add the trigger word)."],
    ["IPAdapter PLUS-FACE weight 0.6 — high enough to hold the face, low enough that pose prompts still move the body.",
     "Wildcards live in custom_nodes/ComfyUI-Impact-Pack/wildcards/ (outfit / pose / angle / framing / expression .txt).",
     "Outfit: signature (fixed OUTFIT) by default; set VARY_OUTFIT=True in config for a swappable-outfit LoRA.",
     "Face detailer uses a pose-NEUTRAL identity prompt so re-rolled crops don't fight the body pose."]),
}


def md(name, g):
    summary, when, steps, knobs = DOCS[name]
    stages = " -> ".join(grp["title"] for grp in sorted(g["groups"], key=lambda gr: gr["bounding"][0]))
    s = [f"# {name}", "", summary, "", f"**When to use:** {when}", "",
         f"**Stages:** {stages}", "", "## How to use"]
    s += [f"{i}. {x}" for i, x in enumerate(steps, 1)]
    s += ["", "## Key knobs"] + [f"- {x}" for x in knobs]

    # every CLIP text-encode node in this workflow, in reading order (x, then y)
    encs = sorted([n for n in g["nodes"] if n["type"] == "CLIPTextEncode"],
                  key=lambda n: (n["pos"][0], n["pos"][1]))
    s += ["", "## Prompts (every CLIP text node in this graph)"]
    for n in encs:
        title = n.get("title", "Prompt")
        feeds, example = PROMPT_INFO.get(title, ("this generation", n["widgets_values"][0]))
        s += [f"- **{title}** — feeds {feeds}.", f"  - example: `{example}`"]
    has_usdu = any(n["type"] == "UltimateSDUpscale" for n in g["nodes"])
    if has_usdu:
        s += ["", "> The **upscaler (UltimateSDUpscale) and face detailer reuse the base "
              "Positive/Negative** above — there is *no* separate upscaler prompt. Only the "
              "hand detailer (and the face-inpaint node in Max) have their own prompt nodes."]

    s += ["", "## Validate", "```",
          f"python tools/validate_workflow.py user/default/workflows/{name}.json", "```", "",
          f"Default checkpoint **{CKPT.split('.')[0]}** (swap in the Checkpoint node). "
          "Fixed seed 1234567890. CLIP skip -2, CFG 5 enforced by the rules file.", "",
          "_Auto-generated by `tools/build_il_graphs.py` — edit there, not here. "
          "Family overview: IL_Graphs_README.md._"]
    return "\n".join(s) + "\n"

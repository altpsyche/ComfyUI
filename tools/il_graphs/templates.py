from __future__ import annotations
import json
from .config import SRC, SEED

if not SRC.exists():
    raise RuntimeError(
        f"harvest source workflow not found: {SRC}\n"
        f"il_graphs harvests node templates from it — run build_il_graphs from the repo root and "
        f"ensure MainGraphv10.json exists under user/default/workflows/.")
_src = json.loads(SRC.read_text(encoding="utf-8"))
TEMPLATES: dict[str, dict] = {}
for _n in _src["nodes"]:
    TEMPLATES.setdefault(_n["type"], _n)


# Hand-authored templates for core nodes absent from the harvest source.
# Schemas verified against ComfyUI's live node definitions (INPUT_TYPES/RETURN_TYPES).
def _io(name, typ, widget=False, links=False):
    d = {"localized_name": name, "name": name, "type": typ}
    if links:
        d["links"] = []
    else:
        d["link"] = None
        if widget:
            d["widget"] = {"name": name}
    return d

def _tpl(ntype, ins, outs, wv, size=(220, 100), cnr="comfy-core"):
    return {"id": 0, "type": ntype, "pos": [0, 0], "size": list(size), "flags": {}, "order": 0, "mode": 0,
            "inputs": ins, "outputs": outs,
            "properties": {"cnr_id": cnr, "Node name for S&R": ntype}, "widgets_values": wv}

EXTRA_TEMPLATES: dict[str, dict] = {
    # ReActor face swap (paste the hero's exact face onto each generated image).
    # Schema from comfyui-reactor-node/nodes.py class `reactor` (ReActorFaceSwap).
    # widgets: [enabled, swap_model, facedetection, face_restore_model, face_restore_visibility,
    #           codeformer_weight, detect_gender_input, detect_gender_source,
    #           input_faces_index, source_faces_index, console_log_level]
    "ReActorFaceSwap": _tpl("ReActorFaceSwap",
        [_io("enabled", "BOOLEAN", widget=True), _io("input_image", "IMAGE"),
         _io("swap_model", "COMBO", widget=True), _io("facedetection", "COMBO", widget=True),
         _io("face_restore_model", "COMBO", widget=True),
         _io("face_restore_visibility", "FLOAT", widget=True), _io("codeformer_weight", "FLOAT", widget=True),
         _io("detect_gender_input", "COMBO", widget=True), _io("detect_gender_source", "COMBO", widget=True),
         _io("input_faces_index", "STRING", widget=True), _io("source_faces_index", "STRING", widget=True),
         _io("console_log_level", "COMBO", widget=True),
         _io("source_image", "IMAGE"), _io("face_model", "FACE_MODEL"), _io("face_boost", "FACE_BOOST")],
        [_io("SWAPPED_IMAGE", "IMAGE", links=True), _io("FACE_MODEL", "FACE_MODEL", links=True),
         _io("ORIGINAL_IMAGE", "IMAGE", links=True)],
        [True, "inswapper_128.onnx", "retinaface_resnet50", "none", 1, 0.5, "no", "no", "0", "0", 1],
        size=(360, 360), cnr="comfyui-reactor-node"),
    # Impact-Pack wildcard prompt encoder (model+clip -> conditioning, expands __wildcards__).
    # Schema read from impact_pack.py ImpactWildcardEncode INPUT_TYPES / RETURN_TYPES.
    "ImpactWildcardEncode": _tpl("ImpactWildcardEncode",
        [_io("model", "MODEL"), _io("clip", "CLIP"),
         _io("wildcard_text", "STRING", widget=True), _io("populated_text", "STRING", widget=True),
         _io("mode", "COMBO", widget=True), _io("Select to add LoRA", "COMBO", widget=True),
         _io("Select to add Wildcard", "COMBO", widget=True), _io("seed", "INT", widget=True)],
        [_io("model", "MODEL", links=True), _io("clip", "CLIP", links=True),
         _io("conditioning", "CONDITIONING", links=True), _io("populated_text", "STRING", links=True)],
        # [wildcard_text, populated_text, mode, Select-LoRA, Select-Wildcard, seed, control_after_generate]
        ["", "", "populate", "Select the LoRA to add to the text",
         "Select the Wildcard to add to the text", SEED, "randomize"],
        size=(400, 280), cnr="ComfyUI-Impact-Pack"),
    # Impact-Pack wildcard PROCESSOR: expands __wildcards__ -> a plain STRING (no model/clip).
    # required: wildcard_text, populated_text, mode, seed(+control), "Select to add Wildcard".
    # NB: `mode` carries a populate/fixed string value ("populate"); the "BOOLEAN" slot-type label
    # below is cosmetic (mode is a pure widget, never linked) and is left as-is so generated graphs
    # stay byte-stable. ImpactWildcardEncode labels the same field "COMBO" — both load fine.
    "ImpactWildcardProcessor": _tpl("ImpactWildcardProcessor",
        [_io("wildcard_text", "STRING", widget=True), _io("populated_text", "STRING", widget=True),
         _io("mode", "BOOLEAN", widget=True), _io("seed", "INT", widget=True),
         _io("Select to add Wildcard", "COMBO", widget=True)],
        [_io("STRING", "STRING", links=True)],
        ["", "", "populate", SEED, "randomize", "Select the Wildcard to add to the text"],
        size=(400, 220), cnr="ComfyUI-Impact-Pack"),

    # ---- Qwen-Image-Edit-2511 (GGUF) edit-dataset nodes. Schemas verified against the live
    # ComfyUI /object_info + the "Image Edit (Qwen 2511)" blueprint (proven to run, 2026-06-15). ----
    # GGUF diffusion loader (ComfyUI-GGUF / city96).
    "UnetLoaderGGUF": _tpl("UnetLoaderGGUF",
        [_io("unet_name", "COMBO", widget=True)],
        [_io("MODEL", "MODEL", links=True)],
        ["qwen-image-edit-2511-Q5_K_M.gguf"], size=(360, 60), cnr="ComfyUI-GGUF"),
    # model-only LoRA loader (stack Lightning + multiple-angles on the GGUF model).
    "LoraLoaderModelOnly": _tpl("LoraLoaderModelOnly",
        [_io("model", "MODEL"), _io("lora_name", "COMBO", widget=True),
         _io("strength_model", "FLOAT", widget=True)],
        [_io("MODEL", "MODEL", links=True)],
        ["", 1.0], size=(340, 82)),
    # text-encoder loader (Qwen 2.5-VL 7B; type 'qwen_image').
    "CLIPLoader": _tpl("CLIPLoader",
        [_io("clip_name", "COMBO", widget=True), _io("type", "COMBO", widget=True),
         _io("device", "COMBO", widget=True)],
        [_io("CLIP", "CLIP", links=True)],
        ["qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", "default"], size=(396, 106)),
    # flow-matching shift + CFG normalization (model patches the blueprint applies for 2511).
    "ModelSamplingAuraFlow": _tpl("ModelSamplingAuraFlow",
        [_io("model", "MODEL"), _io("shift", "FLOAT", widget=True)],
        [_io("MODEL", "MODEL", links=True)], [3.1], size=(270, 58)),
    "CFGNorm": _tpl("CFGNorm",
        [_io("model", "MODEL"), _io("strength", "FLOAT", widget=True)],
        [_io("MODEL", "MODEL", links=True)], [1.0], size=(270, 58)),
    # scale the reference image to the model's optimal pixel budget.
    "FluxKontextImageScale": _tpl("FluxKontextImageScale",
        [_io("image", "IMAGE")], [_io("IMAGE", "IMAGE", links=True)], [], size=(230, 26)),
    # reference-latent method (needed for repackaged/GGUF builds per the blueprint note).
    "FluxKontextMultiReferenceLatentMethod": _tpl("FluxKontextMultiReferenceLatentMethod",
        [_io("conditioning", "CONDITIONING"), _io("reference_latents_method", "COMBO", widget=True)],
        [_io("CONDITIONING", "CONDITIONING", links=True)],
        ["index_timestep_zero"], size=(310, 58)),
    # prompt + reference-image joint encoder (2509/2511). Inputs ordered as the blueprint saves them
    # (clip, vae, image1..3, prompt); vae/image* are optional, prompt accepts a STRING link.
    "TextEncodeQwenImageEditPlus": _tpl("TextEncodeQwenImageEditPlus",
        [_io("clip", "CLIP"), _io("vae", "VAE"), _io("image1", "IMAGE"),
         _io("image2", "IMAGE"), _io("image3", "IMAGE"), _io("prompt", "STRING", widget=True)],
        [_io("CONDITIONING", "CONDITIONING", links=True)],
        [""], size=(420, 200), cnr="comfy-core"),

    # ---- efficiency-nodes (jags111) — used by IL_XYPlot to grid LoRA epochs x strength.
    # Schemas read from custom_nodes/efficiency-nodes-comfyui/efficiency_nodes.py INPUT/RETURN. ----
    # all-in-one loader (ckpt + vae + clip-skip + prompt encode + empty latent). widgets:
    # [ckpt, vae, clip_skip, lora_name, lora_model_str, lora_clip_str, positive, negative,
    #  token_normalization, weight_interpretation, latent_w, latent_h, batch_size]
    "Efficient Loader": _tpl("Efficient Loader",
        [_io("ckpt_name", "COMBO", widget=True), _io("vae_name", "COMBO", widget=True),
         _io("clip_skip", "INT", widget=True), _io("lora_name", "COMBO", widget=True),
         _io("lora_model_strength", "FLOAT", widget=True), _io("lora_clip_strength", "FLOAT", widget=True),
         _io("positive", "STRING", widget=True), _io("negative", "STRING", widget=True),
         _io("token_normalization", "COMBO", widget=True), _io("weight_interpretation", "COMBO", widget=True),
         _io("empty_latent_width", "INT", widget=True), _io("empty_latent_height", "INT", widget=True),
         _io("batch_size", "INT", widget=True),
         _io("lora_stack", "LORA_STACK"), _io("cnet_stack", "CONTROL_NET_STACK")],
        [_io("MODEL", "MODEL", links=True), _io("CONDITIONING+", "CONDITIONING", links=True),
         _io("CONDITIONING-", "CONDITIONING", links=True), _io("LATENT", "LATENT", links=True),
         _io("VAE", "VAE", links=True), _io("CLIP", "CLIP", links=True),
         _io("DEPENDENCIES", "DEPENDENCIES", links=True)],
        ["model.safetensors", "Baked VAE", -2, "None", 1.0, 1.0, "positive", "negative",
         "none", "comfy", 832, 1216, 1],
        size=(400, 320), cnr="efficiency-nodes-comfyui"),
    # KSampler with an optional `script` input (the XY Plot SCRIPT plugs in here). seed widget carries a
    # control_after_generate entry. widgets: [seed, control, steps, cfg, sampler, scheduler, denoise,
    # preview_method, vae_decode]
    "KSampler (Efficient)": _tpl("KSampler (Efficient)",
        [_io("model", "MODEL"), _io("seed", "INT", widget=True), _io("steps", "INT", widget=True),
         _io("cfg", "FLOAT", widget=True), _io("sampler_name", "COMBO", widget=True),
         _io("scheduler", "COMBO", widget=True), _io("positive", "CONDITIONING"),
         _io("negative", "CONDITIONING"), _io("latent_image", "LATENT"), _io("denoise", "FLOAT", widget=True),
         _io("preview_method", "COMBO", widget=True), _io("vae_decode", "COMBO", widget=True),
         _io("optional_vae", "VAE"), _io("script", "SCRIPT")],
        [_io("MODEL", "MODEL", links=True), _io("CONDITIONING+", "CONDITIONING", links=True),
         _io("CONDITIONING-", "CONDITIONING", links=True), _io("LATENT", "LATENT", links=True),
         _io("VAE", "VAE", links=True), _io("IMAGE", "IMAGE", links=True)],
        [0, "fixed", 20, 7.0, "euler", "normal", 1.0, "auto", "true"],
        size=(340, 360), cnr="efficiency-nodes-comfyui"),
    # XY Plot script: combines the X/Y axes (+ loader DEPENDENCIES) into a SCRIPT for the Efficient KSampler.
    "XY Plot": _tpl("XY Plot",
        [_io("grid_spacing", "INT", widget=True), _io("XY_flip", "COMBO", widget=True),
         _io("Y_label_orientation", "COMBO", widget=True), _io("cache_models", "COMBO", widget=True),
         _io("ksampler_output_image", "COMBO", widget=True),
         _io("dependencies", "DEPENDENCIES"), _io("X", "XY"), _io("Y", "XY")],
        [_io("SCRIPT", "SCRIPT", links=True)],
        [10, "False", "Horizontal", "True", "Plot"], size=(320, 200), cnr="efficiency-nodes-comfyui"),
    # XY axis source: a batch of LoRA files (X) crossed with a weight sweep (Y). Outputs BOTH X and Y.
    # widgets: [input_mode, lora_name, model_str, clip_str, X_batch_count, X_batch_path, X_subdirs,
    #  X_batch_sort, X_first, X_last, Y_batch_count, Y_first, Y_last]
    "XY Input: LoRA Plot": _tpl("XY Input: LoRA Plot",
        [_io("input_mode", "COMBO", widget=True), _io("lora_name", "COMBO", widget=True),
         _io("model_strength", "FLOAT", widget=True), _io("clip_strength", "FLOAT", widget=True),
         _io("X_batch_count", "INT", widget=True), _io("X_batch_path", "STRING", widget=True),
         _io("X_subdirectories", "BOOLEAN", widget=True), _io("X_batch_sort", "COMBO", widget=True),
         _io("X_first_value", "FLOAT", widget=True), _io("X_last_value", "FLOAT", widget=True),
         _io("Y_batch_count", "INT", widget=True), _io("Y_first_value", "FLOAT", widget=True),
         _io("Y_last_value", "FLOAT", widget=True), _io("lora_stack", "LORA_STACK")],
        [_io("X", "XY", links=True), _io("Y", "XY", links=True)],
        ["X: LoRA Batch, Y: LoRA Weight", "None", 1.0, 1.0, 3, "", False, "ascending",
         0.0, 1.0, 3, 0.5, 0.9], size=(400, 360), cnr="efficiency-nodes-comfyui"),
}
for _k, _v in EXTRA_TEMPLATES.items():
    TEMPLATES.setdefault(_k, _v)

from __future__ import annotations
import json
from .config import SRC, SEED

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
}
for _k, _v in EXTRA_TEMPLATES.items():
    TEMPLATES.setdefault(_k, _v)

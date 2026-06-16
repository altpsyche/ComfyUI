from __future__ import annotations
from .builder import Builder
from .config import (CKPT, VAE, SEED, NEG, REF_SUFFIX,
                     BASE_SAMPLER, BASE_SCHED, BASE_STEPS, BASE_CFG, NOTE_C, NOTE_BG)
from .layers import (core, add_upscale, add_detailers, add_face_inpaint,
                     add_bg, add_finish)


def build_base():
    b = Builder(); c = core(b)
    add_finish(b, (c["dec"], "IMAGE"), "IL_1_Base", x=-760)
    return b.build()

def build_refine():
    b = Builder(); c = core(b)
    u = add_upscale(b, c, with_cn=False)
    add_finish(b, u, "IL_2_Refine", x=540)
    return b.build()

def build_guided():
    b = Builder(); c = core(b)
    u = add_upscale(b, c, with_cn=False)
    h = add_detailers(b, c, u)
    add_finish(b, h, "IL_3_Guided", x=1700)
    return b.build()

def build_studio():
    b = Builder(); c = core(b)
    u = add_upscale(b, c, with_cn=True)
    h = add_detailers(b, c, u)
    bg = add_bg(b, c, h)
    add_finish(b, bg, "IL_4_Studio", x=3600, metadata=True, aesthetic=True)
    return b.build()

def build_max():
    b = Builder(); c = core(b)
    u = add_upscale(b, c, with_cn=True)
    h = add_detailers(b, c, u)
    fi = add_face_inpaint(b, c, h)
    bg = add_bg(b, c, fi)
    add_finish(b, bg, "IL_5_Max", x=3600, sharpen=True, metadata=True, aesthetic=True)
    return b.build()


# ---- dedicated feature graphs (orthogonal to the ladder; feature is ACTIVE) ----
def build_ipadapter():
    b = Builder(); c = core(b, ipadapter=True)
    u = add_upscale(b, c, with_cn=False)
    h = add_detailers(b, c, u)
    add_finish(b, h, "IL_IPAdapter", x=1700, metadata=True)
    return b.build()

def build_pose():
    b = Builder(); c = core(b, pose=True)
    u = add_upscale(b, c, with_cn=False)
    h = add_detailers(b, c, u)
    add_finish(b, h, "IL_Pose", x=1700, metadata=True)
    return b.build()

def build_lcm():
    b = Builder(); c = core(b, lcm=True)   # fast preview: base only, no heavy post
    add_finish(b, (c["dec"], "IMAGE"), "IL_LCM", x=-760)
    return b.build()


# Qwen-Image-Edit-2511 model stack (downloaded by scripts/install_qwen_edit.ps1).
QE_GGUF = "qwen-image-edit-2511-Q5_K_M.gguf"
QE_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QE_VAE = "qwen_image_vae.safetensors"
QE_LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QE_ANGLES = "qwen-image-edit-2511-multiple-angles-lora.safetensors"


def build_dataset_edit(name="edit", identity="1girl, solo", outfit=""):
    """FRONTIER dataset generator (self-contained): generate ONE original hero, then re-pose it into
    many varied shots with Qwen-Image-Edit-2511 (GGUF), holding identity + art style. One graph per
    roster character (IL_DatasetEdit_<name>); the trainer reads output/dataset/<name>/ as usual.

    Stage 1 (bootstrap the hero): an Illustrious text2img renders the character from its `id` tags at
    a fixed Hero Seed -> HERO preview. Reroll the Hero Seed to pick a face you like; that single image
    is the identity anchor. Stage 2 (propagate): FluxKontextImageScale -> TextEncodeQwenImageEditPlus
    (scaled hero + a wildcard instruction) -> KSampler on the GGUF model (Lightning 4-step, 6 steps /
    cfg 1.0) -> save to output/dataset/<name>/. Because the hero is rendered in YOUR checkpoint, the
    edits stay on-style; the edit only changes pose/angle/expression. Verified on the live ComfyUI.

    Layout: clean left->right columns (Stage1 hero | model | encoders | instruction | refs | edit).
    """
    b = Builder()
    # ===== STAGE 1 — Illustrious hero generator (text2img from the character's id; pick a face) =====
    ck = b.add("CheckpointLoaderSimple", [CKPT], pos=(0, 0), title="Checkpoint (Illustrious)")
    hvae = b.add("VAELoader", [VAE], pos=(0, 160), title="VAE (hero)")
    hseed = b.add("Seed", [SEED, "fixed"], pos=(0, 300), title="Hero Seed (fixed = pick the face)")
    hclip = b.add("CLIPSetLastLayer", [-2], pos=(0, 450), title="CLIP skip -2")
    b.link(ck, "CLIP", hclip, "clip")
    hpos = b.add("CLIPTextEncode", [identity + (", " + outfit if outfit else "") + REF_SUFFIX],
                 pos=(360, 0), title="Hero prompt (identity)")
    hneg = b.add("CLIPTextEncode", [NEG], pos=(360, 180), title="Negative")
    hlat = b.add("EmptyLatentImage", [832, 1216, 1], pos=(360, 360), title="Hero latent 832x1216")
    b.link(hclip, "CLIP", hpos, "clip"); b.link(hclip, "CLIP", hneg, "clip")
    hks = b.add("KSampler", [SEED, "fixed", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0],
                pos=(720, 0), title="Hero KSampler")
    b.link(ck, "MODEL", hks, "model"); b.link(hpos, "CONDITIONING", hks, "positive")
    b.link(hneg, "CONDITIONING", hks, "negative"); b.link(hlat, "LATENT", hks, "latent_image")
    b.link(hseed, "int", hks, "seed")
    hdec = b.add("VAEDecode", [], pos=(720, 320), title="Hero decode")
    b.link(hks, "LATENT", hdec, "samples"); b.link(hvae, "VAE", hdec, "vae")
    hprev = b.add("PreviewImage", [], pos=(720, 470), title="HERO preview (reroll Hero Seed to pick the face)")
    b.link(hdec, "IMAGE", hprev, "images")

    # ===== STAGE 2 model — GGUF + Lightning(+angles) LoRA -> flow-shift + CFGNorm (2511 patch chain) =====
    gguf = b.add("UnetLoaderGGUF", [QE_GGUF], pos=(1180, 0), title="Qwen-Edit GGUF (Q5)")
    llora = b.add("LoraLoaderModelOnly", [QE_LIGHTNING, 1.0], pos=(1180, 150), title="Lightning 4-step LoRA")
    alora = b.add("LoraLoaderModelOnly", [QE_ANGLES, 1.0], pos=(1180, 300), title="Multiple-angles LoRA")
    msaf = b.add("ModelSamplingAuraFlow", [3.1], pos=(1180, 450), title="ModelSampling (shift 3.1)")
    cfgn = b.add("CFGNorm", [1.0], pos=(1180, 600), title="CFGNorm")
    b.link(gguf, "MODEL", llora, "model"); b.link(llora, "MODEL", alora, "model")
    b.link(alora, "MODEL", msaf, "model"); b.link(msaf, "MODEL", cfgn, "model")

    # ===== encoders + scale =====
    clip = b.add("CLIPLoader", [QE_CLIP, "qwen_image", "default"], pos=(1620, 0), title="Qwen 2.5-VL text encoder")
    vae = b.add("VAELoader", [QE_VAE], pos=(1620, 170), title="Qwen Image VAE")
    scale = b.add("FluxKontextImageScale", [], pos=(1620, 340), title="Scale ref (hero -> edit)")
    b.link(hdec, "IMAGE", scale, "image")   # the Stage-1 hero feeds the edit

    # ===== instruction + encode (wildcards vary framing/angle/pose/expr/background/lighting) =====
    # CRITICAL (variety fix): the ImpactWildcardProcessor BACKEND only expands wildcards found in
    # populated_text (doit() -> process(populated_text, seed)); the "populate" copy of wildcard_text
    # -> populated_text is a browser-JS step that NEVER runs headless (API POST) and may fire only
    # once per Queue in the UI. So we put the SAME wildcard string in BOTH boxes and set mode "fixed":
    # the backend then re-rolls a fresh instruction every execution, keyed on the seed (which
    # control_after_generate=randomize advances per batch item in the UI / per POST via the runner).
    # Phrasing leads with the imperative CHANGE because Qwen-Image-Edit is conservative; the identity
    # + style lock is a concise trailing clause so it doesn't drown the edit verbs.
    wtext = ("Change the shot to __framing__ from __angle__. Re-pose the character to __pose__, __expression__. "
             "Set the scene: __background__, __lighting__. "
             "Keep the exact same character (identical face, hairstyle and outfit) and the same anime art style.")
    wild = b.add("ImpactWildcardProcessor",
                 [wtext, wtext, "fixed", SEED, "randomize", "Select the Wildcard to add to the text"],
                 pos=(2120, 0), title="Edit instruction (re-rolls every run)")
    posenc = b.add("TextEncodeQwenImageEditPlus", [""], pos=(2120, 220), title="Encode (positive: hero + instruction)")
    negenc = b.add("TextEncodeQwenImageEditPlus", [""], pos=(2120, 470), title="Encode (negative: empty)")
    for enc in (posenc, negenc):
        b.link(clip, "CLIP", enc, "clip"); b.link(vae, "VAE", enc, "vae"); b.link(scale, "IMAGE", enc, "image1")
    b.link(wild, "STRING", posenc, "prompt")   # instruction drives the positive encoder

    # ===== reference-latent method (needed for repackaged/GGUF builds) + init latent =====
    posref = b.add("FluxKontextMultiReferenceLatentMethod", ["index_timestep_zero"], pos=(2660, 220), title="Ref method (pos)")
    negref = b.add("FluxKontextMultiReferenceLatentMethod", ["index_timestep_zero"], pos=(2660, 470), title="Ref method (neg)")
    b.link(posenc, "CONDITIONING", posref, "conditioning"); b.link(negenc, "CONDITIONING", negref, "conditioning")
    venc = b.add("VAEEncode", [], pos=(2660, 640), title="Encode hero -> latent")
    b.link(scale, "IMAGE", venc, "pixels"); b.link(vae, "VAE", venc, "vae")

    # ===== edit + decode (Lightning: 6 steps / cfg 1.0 / euler / simple) =====
    ks = b.add("KSampler", [SEED, "randomize", 6, 1.0, "euler", "simple", 1.0], pos=(3120, 220), title="Edit KSampler (6 steps)")
    b.link(cfgn, "MODEL", ks, "model"); b.link(posref, "CONDITIONING", ks, "positive")
    b.link(negref, "CONDITIONING", ks, "negative"); b.link(venc, "LATENT", ks, "latent_image")
    vdec = b.add("VAEDecode", [], pos=(3120, 460), title="Decode")
    b.link(ks, "LATENT", vdec, "samples"); b.link(vae, "VAE", vdec, "vae")

    note = b.add("Note", [
        f"QWEN-EDIT DATASET TOOL ('{name}') -- self-contained: it MAKES the hero, then re-poses it.\n"
        "STAGE 1 (left): reroll 'Hero Seed' and watch HERO preview until you like the face (rendered\n"
        "from this character's id tags in YOUR checkpoint). Then leave Hero Seed fixed on that value.\n"
        "STAGE 2: leave 'Edit instruction' seed control = randomize; set batch count ~40 + Queue once\n"
        "  -> output/dataset/" + name + "/. Each frame = that hero in a new framing/angle/pose/expression/\n"
        "  background/lighting, same identity + art style. (mode is 'fixed' on purpose: the wildcards\n"
        "  expand in the node backend every run, keyed on the seed -- so it varies headless too.)\n"
        "Then curate the best 25-40 and run: tools/lora_train/train_lora.ps1 -Char " + name + ".\n"
        "Wildcards (__framing__/__angle__/__pose__/__expression__/__background__/__lighting__) live in\n"
        "  ComfyUI-Impact-Pack/wildcards/. Too slow / OOM? re-run install_qwen_edit.ps1 -Quant Q4_K_M.\n"
        "Identity drifting? lower the multiple-angles LoRA strength or trim the instruction's scene clause."],
        pos=(2120, 700), title=f"How to use ({name})", color=NOTE_C, bgcolor=NOTE_BG)

    add_finish(b, (vdec, "IMAGE"), f"dataset/{name}/{name}", x=3520)
    b.group("STAGE 1 - Hero generator (Illustrious)", [ck, hvae, hseed, hclip, hpos, hneg, hlat, hks, hdec, hprev], "#535")
    b.group("STAGE 2 - Qwen-Edit model + LoRAs", [gguf, llora, alora, msaf, cfgn], "#525")
    b.group("Encoders + scale", [clip, vae, scale], "#535")
    b.group("Instruction + encode", [wild, posenc, negenc, note], "#355")
    b.group("Reference + latent", [posref, negref, venc], "#355")
    b.group("Edit + decode", [ks, vdec], "#553")
    return b.build()

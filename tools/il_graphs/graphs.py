from __future__ import annotations
import re
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


def _esc(t):
    """Escape unescaped ( ) so the CLIP encoder doesn't read a danbooru series suffix
    (e.g. 'ganyu (genshin impact)') as prompt-weighting syntax."""
    return re.sub(r"(?<!\\)([()])", r"\\\1", t)


def build_dataset(name, identity, outfit, vary_outfit=False, base=""):
    """Synthetic training-data generator for ONE roster character (one graph per CHARACTERS entry).

    Two identity-consistency modes:
      base SET    pure-text path -- a known danbooru character tag carries the face, so the
                  hero+IPAdapter scaffold is OFF; the base render alone stays on-model.
      base EMPTY  a fixed-seed HERO portrait feeds a LIGHT IPAdapter PLUS-FACE that pins that face
                  onto the face pass only (the original-face route).
    Either way an Impact wildcard prompt varies (outfit)/pose/angle/framing/expression and the Gen
    Seed re-rolls. Curate the on-model outputs to ~30, then train_lora.ps1 -Char <name>. No external
    image loads.
    """
    text_only = bool((base or "").strip())
    ident = f"{_esc(base)}, {identity}" if text_only else identity

    b = Builder()
    ck = b.add("CheckpointLoaderSimple", [CKPT], pos=(-2700, -100), title="Checkpoint")
    vae = b.add("VAELoader", [VAE], pos=(-2700, 60), title="VAE")
    if not text_only:
        hseed = b.add("Seed", [SEED, "fixed"], pos=(-2700, 220), title="Hero Seed (fixed = same face)")
    gseed = b.add("Seed", [SEED, "randomize"], pos=(-2700, 360), title="Gen Seed (reroll = variety)")
    clip = b.add("CLIPSetLastLayer", [-2], pos=(-2380, -100), title="CLIP skip -2")
    b.link(ck, "CLIP", clip, "clip")
    neg = b.add("CLIPTextEncode", [NEG], pos=(-2060, 320), title="Negative")
    b.link(clip, "CLIP", neg, "clip")

    if not text_only:
        # --- HERO portrait (in-graph, fixed seed): the single identity source ---
        # Hero always wears the fixed outfit (it's only the face source; IPAdapter is face-only anyway).
        hpos = b.add("CLIPTextEncode", [identity + ", " + outfit + REF_SUFFIX], pos=(-2060, -200), title="Hero portrait prompt")
        b.link(clip, "CLIP", hpos, "clip")
        hlat = b.add("EmptyLatentImage", [832, 1216, 1], pos=(-2060, 60), title="Hero latent 832x1216")
        hks = b.add("KSampler", [SEED, "fixed", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0],
                    pos=(-1720, -200), title="Hero KSampler")
        b.link(ck, "MODEL", hks, "model"); b.link(hpos, "CONDITIONING", hks, "positive")
        b.link(neg, "CONDITIONING", hks, "negative"); b.link(hlat, "LATENT", hks, "latent_image")
        b.link(hseed, "int", hks, "seed")
        hdec = b.add("VAEDecode", [], pos=(-1400, -200), title="Hero decode")
        b.link(hks, "LATENT", hdec, "samples"); b.link(vae, "VAE", hdec, "vae")
        hprev = b.add("PreviewImage", [], pos=(-1400, -460), title="HERO preview (reroll Hero Seed to pick the face)")
        b.link(hdec, "IMAGE", hprev, "images")

        # --- IPAdapter PLUS-FACE: a hero-identity model used ONLY by the face pass, at MODERATE weight
        # + "V only" so the wildcard expression/pose still come through (strong K+V froze expressions;
        # ReActor froze them harder + looked uncanny). Text-driven: the hero itself is text2img. ---
        ipl = b.add("IPAdapterUnifiedLoader", ["PLUS FACE (portraits)"], pos=(-1400, 60), title="IPAdapter loader")
        ipa = b.add("IPAdapterAdvanced", [0.55, "ease in-out", "concat", 0, 1, "V only"],
                    pos=(-1400, 230), title="IPAdapter apply (face, 0.55 V-only)")
        b.link(ck, "MODEL", ipl, "model")
        b.link(ipl, "model", ipa, "model"); b.link(ipl, "ipadapter", ipa, "ipadapter")
        b.link(hdec, "IMAGE", ipa, "image")

    # --- variation prompt: identity + outfit + wildcards (reroll Gen Seed to repopulate) ---
    # vary_outfit True -> __outfit__ wildcard (swappable-outfit LoRA); False -> fixed outfit (signature).
    outfit_tok = "__outfit__" if vary_outfit else outfit
    wtext = ident + ", " + outfit_tok + ", __framing__, __angle__, __pose__, __expression__"
    populated = ident + ", " + outfit + ", upper body, front view, standing, neutral expression"
    we = b.add("ImpactWildcardEncode",
               [wtext, populated, "populate", "Select the LoRA to add to the text",
                "Select the Wildcard to add to the text", SEED, "randomize"],
               pos=(-1060, -100), title="Wildcard prompt (outfit/pose/angle/framing/expr)")
    # base render uses the RAW checkpoint (clean, crisp); identity comes from the tag (base mode) or
    # is applied lightly at the face pass via IPAdapter (hero mode).
    b.link(ck, "MODEL", we, "model"); b.link(clip, "CLIP", we, "clip")

    # --- batched generation (varying seed) ---
    mlat = b.add("EmptyLatentImage", [1024, 1024, 4], pos=(-1060, 220), title="Batch latent 1024 x4")
    mks = b.add("KSampler", [SEED, "randomize", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0],
                pos=(-720, -100), title="Gen KSampler (batch 4)")
    b.link(we, "model", mks, "model"); b.link(we, "conditioning", mks, "positive")
    b.link(neg, "CONDITIONING", mks, "negative"); b.link(mlat, "LATENT", mks, "latent_image")
    b.link(gseed, "int", mks, "seed")
    mdec = b.add("VAEDecode", [], pos=(-400, -100), title="Gen decode")
    b.link(mks, "LATENT", mdec, "samples"); b.link(vae, "VAE", mdec, "vae")

    # --- clean the face + hands on the RAW model (a good swap target), pose-NEUTRAL identity cond ---
    nface = b.add("CLIPTextEncode", [ident], pos=(-400, 220), title="Face detail (neutral identity)")
    b.link(clip, "CLIP", nface, "clip")
    c = dict(msrc=(we, "model"), clip=clip, vae=vae,   # msrc = raw checkpoint (clean render throughout)
             cpos=(we, "conditioning"), cneg=(neg, "CONDITIONING"), seed=gseed)
    # base mode: face pass on the raw model (the tag holds identity), standard denoise. hero mode:
    # face pass on the IPAdapter model at a slightly higher denoise to impose the hero face lightly.
    face_model = None if text_only else (ipa, "MODEL")
    face_denoise = 0.3 if text_only else 0.4
    h = add_detailers(b, c, (mdec, "IMAGE"), x=-60, face_cond=(nface, "CONDITIONING"),
                      face_model=face_model, face_denoise=face_denoise)

    if text_only:
        note_text = (
            f"DATASET TOOL for character '{name}' -- BASE / TEXT-ONLY mode (base tag set in config).\n"
            f"IPAdapter OFF: the danbooru tag '{base}' carries the face; pure text keeps it consistent.\n"
            "Expressions/poses/framing still VARY via wildcards. No hero, no IPAdapter, no face-swap.\n"
            f"1. Reroll Gen Seed (batch of 4) -> ~60 varied shots into output/dataset/{name}/.\n"
            "2. Delete the off-model ones IN PLACE (curation) -- keep the best 25-40.\n"
            f"3. Train:  tools/lora_train/train_lora.ps1 -Char {name}   (or train_all.ps1 for the roster)\n"
            "The trained LoRA -- not the dataset -- gives the final exact face. Edit CHARACTERS in config.py.\n"
            "Face drifting? pick a more iconic danbooru base tag, or keep id minimal so the tag dominates.")
    else:
        note_text = (
            f"DATASET TOOL for character '{name}' (one graph per CHARACTERS entry in config).\n"
            "TEXT-DRIVEN: identity comes from the prompt tags; the in-graph hero + a LIGHT IPAdapter keep\n"
            "the face consistent while expressions/poses still VARY. No external images, no face-swap.\n"
            "1. Hero Seed fixed; pick a hero in HERO preview (the identity anchor).\n"
            f"2. Reroll Gen Seed (batch of 4) -> ~60 varied shots into output/dataset/{name}/.\n"
            "3. Delete the off-model ones IN PLACE (curation) -- keep the best 25-40.\n"
            f"4. Train:  tools/lora_train/train_lora.ps1 -Char {name}   (or train_all.ps1 for the roster)\n"
            "The trained LoRA -- not the dataset -- gives the final exact face. Edit CHARACTERS in config.py.\n"
            "Face too inconsistent? raise IPAdapter weight. Too samey/stiff? lower it.")
    note = b.add("Note", [note_text],
        pos=(-1060, 520), title=f"How to use ({name})", color=NOTE_C, bgcolor=NOTE_BG)

    # SaveImage splits the prefix on the LAST "/": "dataset/<name>/<name>" => a real per-character
    # subfolder output/dataset/<name>/ (just "dataset/<name>" would dump every character into output/dataset/).
    add_finish(b, h, f"dataset/{name}/{name}", x=1100)
    load_seed_nodes = [ck, vae, gseed, clip, neg] if text_only else [ck, vae, hseed, gseed, clip, neg]
    b.group("Load + Seeds", load_seed_nodes, "#535")
    if not text_only:
        b.group("Hero portrait (identity source)", [hpos, hlat, hks, hdec, hprev], "#525")
        b.group("IPAdapter face lock (light)", [ipl, ipa], "#525")
    b.group("Variation prompt", [we, mlat, note], "#355")
    b.group("Batched generation", [mks, mdec, nface], "#553")
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
    alora = b.add("LoraLoaderModelOnly", [QE_ANGLES, 0.8], pos=(1180, 300), title="Multiple-angles LoRA")
    msaf = b.add("ModelSamplingAuraFlow", [3.1], pos=(1180, 450), title="ModelSampling (shift 3.1)")
    cfgn = b.add("CFGNorm", [1.0], pos=(1180, 600), title="CFGNorm")
    b.link(gguf, "MODEL", llora, "model"); b.link(llora, "MODEL", alora, "model")
    b.link(alora, "MODEL", msaf, "model"); b.link(msaf, "MODEL", cfgn, "model")

    # ===== encoders + scale =====
    clip = b.add("CLIPLoader", [QE_CLIP, "qwen_image", "default"], pos=(1620, 0), title="Qwen 2.5-VL text encoder")
    vae = b.add("VAELoader", [QE_VAE], pos=(1620, 170), title="Qwen Image VAE")
    scale = b.add("FluxKontextImageScale", [], pos=(1620, 340), title="Scale ref (hero -> edit)")
    b.link(hdec, "IMAGE", scale, "image")   # the Stage-1 hero feeds the edit

    # ===== instruction + encode (wildcards vary pose/angle/expression; identity held by the ref) =====
    wtext = ("same character, identical face and hair and outfit, keep the same art style, "
             "__angle__, __pose__, __expression__")
    populated = ("same character, identical face and hair and outfit, keep the same art style, "
                 "side view, sitting, neutral expression")
    wild = b.add("ImpactWildcardProcessor",
                 [wtext, populated, "populate", SEED, "randomize", "Select the Wildcard to add to the text"],
                 pos=(2120, 0), title="Edit instruction (reroll = variety)")
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
        "STAGE 2: reroll the 'Edit instruction' seed (batch-queue ~40) -> output/dataset/" + name + "/.\n"
        "  Each frame = that hero in a new angle/pose/expression, same identity + art style.\n"
        "Then curate the best 25-40 and run: tools/lora_train/train_lora.ps1 -Char " + name + ".\n"
        "Wildcards (__angle__/__pose__/__expression__) live in ComfyUI-Impact-Pack/wildcards/.\n"
        "Too slow / OOM? re-run install_qwen_edit.ps1 -Quant Q4_K_M. Identity drifting? lower the\n"
        "multiple-angles LoRA strength or simplify the instruction."],
        pos=(2120, 700), title=f"How to use ({name})", color=NOTE_C, bgcolor=NOTE_BG)

    add_finish(b, (vdec, "IMAGE"), f"dataset/{name}/{name}", x=3520)
    b.group("STAGE 1 - Hero generator (Illustrious)", [ck, hvae, hseed, hclip, hpos, hneg, hlat, hks, hdec, hprev], "#535")
    b.group("STAGE 2 - Qwen-Edit model + LoRAs", [gguf, llora, alora, msaf, cfgn], "#525")
    b.group("Encoders + scale", [clip, vae, scale], "#535")
    b.group("Instruction + encode", [wild, posenc, negenc, note], "#355")
    b.group("Reference + latent", [posref, negref, venc], "#355")
    b.group("Edit + decode", [ks, vdec], "#553")
    return b.build()

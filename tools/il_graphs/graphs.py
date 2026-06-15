from __future__ import annotations
from .builder import Builder
from .config import (CKPT, VAE, SEED, NEG, CHAR, CHAR_NAME, REF_SUFFIX,
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


def build_dataset():
    """Synthetic training-data generator for ONE character (edit CHAR in config).

    A fixed-seed HERO portrait is the only identity source: it feeds an IPAdapter PLUS-FACE
    that pins that face onto every render, while an Impact wildcard prompt varies
    pose / angle / framing / expression and the Gen Seed re-rolls. Curate the on-model
    outputs into a ~30-image set, caption, and train a LoRA. No external image loads.
    """
    b = Builder()
    ck = b.add("CheckpointLoaderSimple", [CKPT], pos=(-2700, -100), title="Checkpoint")
    vae = b.add("VAELoader", [VAE], pos=(-2700, 60), title="VAE")
    hseed = b.add("Seed", [SEED, "fixed"], pos=(-2700, 220), title="Hero Seed (fixed = same face)")
    gseed = b.add("Seed", [SEED, "randomize"], pos=(-2700, 360), title="Gen Seed (reroll = variety)")
    clip = b.add("CLIPSetLastLayer", [-2], pos=(-2380, -100), title="CLIP skip -2")
    b.link(ck, "CLIP", clip, "clip")
    neg = b.add("CLIPTextEncode", [NEG], pos=(-2060, 320), title="Negative")
    b.link(clip, "CLIP", neg, "clip")

    # --- HERO portrait (in-graph, fixed seed): the single identity source ---
    hpos = b.add("CLIPTextEncode", [CHAR + REF_SUFFIX], pos=(-2060, -200), title="Hero portrait prompt")
    b.link(clip, "CLIP", hpos, "clip")
    hlat = b.add("EmptyLatentImage", [832, 1216, 1], pos=(-2060, 60), title="Hero latent 832x1216")
    hks = b.add("KSampler", [SEED, "fixed", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0],
                pos=(-1720, -200), title="Hero KSampler")
    b.link(ck, "MODEL", hks, "model"); b.link(hpos, "CONDITIONING", hks, "positive")
    b.link(neg, "CONDITIONING", hks, "negative"); b.link(hlat, "LATENT", hks, "latent_image")
    b.link(hseed, "int", hks, "seed")
    hdec = b.add("VAEDecode", [], pos=(-1400, -200), title="Hero decode")
    b.link(hks, "LATENT", hdec, "samples"); b.link(vae, "VAE", hdec, "vae")

    # --- IPAdapter PLUS-FACE: pin the hero face (weight 0.6 leaves room for pose variety) ---
    ipl = b.add("IPAdapterUnifiedLoader", ["PLUS FACE (portraits)"], pos=(-1400, 60), title="IPAdapter loader")
    ipa = b.add("IPAdapterAdvanced", [0.6, "ease in-out", "concat", 0, 1, "V only"],
                pos=(-1400, 230), title="IPAdapter apply (face, 0.6)")
    b.link(ck, "MODEL", ipl, "model")
    b.link(ipl, "model", ipa, "model"); b.link(ipl, "ipadapter", ipa, "ipadapter")
    b.link(hdec, "IMAGE", ipa, "image")

    # --- variation prompt: identity tags + wildcards (reroll Gen Seed to repopulate) ---
    wtext = CHAR + ", __framing__, __angle__, __pose__, __expression__"
    populated = CHAR + ", upper body, front view, standing, neutral expression"
    we = b.add("ImpactWildcardEncode",
               [wtext, populated, "populate", "Select the LoRA to add to the text",
                "Select the Wildcard to add to the text", SEED, "randomize"],
               pos=(-1060, -100), title="Wildcard prompt (pose/angle/framing/expr)")
    b.link(ipa, "MODEL", we, "model"); b.link(clip, "CLIP", we, "clip")

    # --- batched generation (varying seed) ---
    mlat = b.add("EmptyLatentImage", [1024, 1024, 4], pos=(-1060, 220), title="Batch latent 1024 x4")
    mks = b.add("KSampler", [SEED, "randomize", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0],
                pos=(-720, -100), title="Gen KSampler (batch 4)")
    b.link(we, "model", mks, "model"); b.link(we, "conditioning", mks, "positive")
    b.link(neg, "CONDITIONING", mks, "negative"); b.link(mlat, "LATENT", mks, "latent_image")
    b.link(gseed, "int", mks, "seed")
    mdec = b.add("VAEDecode", [], pos=(-400, -100), title="Gen decode")
    b.link(mks, "LATENT", mdec, "samples"); b.link(vae, "VAE", mdec, "vae")

    # --- crisp faces for training (pose-NEUTRAL face cond, per the detailer rule) ---
    nface = b.add("CLIPTextEncode", [CHAR], pos=(-400, 220), title="Face detail (neutral identity)")
    b.link(clip, "CLIP", nface, "clip")
    c = dict(msrc=(we, "model"), clip=clip, vae=vae,
             cpos=(we, "conditioning"), cneg=(neg, "CONDITIONING"), seed=gseed)
    h = add_detailers(b, c, (mdec, "IMAGE"), x=-60, face_cond=(nface, "CONDITIONING"))

    note = b.add("Note", [
        "DATASET TOOL — build a character-LoRA training set. One character per run.\n"
        "1. Set the character: edit the 'Character identity' prompt + the SaveImage prefix\n"
        "   ('dataset/<name>') here in the UI -- no regen, no file moving. Images for each\n"
        "   character land in their own output/dataset/<name>/ folder.\n"
        "2. Hero Seed fixed; pick a hero portrait you like (it sets the face).\n"
        "3. Reroll Gen Seed (batch of 4) -> ~60 varied shots straight into that folder.\n"
        "4. Delete the off-model ones IN PLACE (curation), then run:\n"
        "     tools/lora_train/train_lora.ps1 -Char <name>   (auto-captions + trains)\n"
        "Wildcards: pose/angle/framing/expression .txt in ComfyUI-Impact-Pack/wildcards/."],
        pos=(-1060, 500), title="How to use", color=NOTE_C, bgcolor=NOTE_BG)

    add_finish(b, h, f"dataset/{CHAR_NAME}", x=1100)
    b.group("Load + Seeds", [ck, vae, hseed, gseed, clip, neg], "#535")
    b.group("Hero portrait (identity source)", [hpos, hlat, hks, hdec], "#525")
    b.group("IPAdapter face lock", [ipl, ipa], "#525")
    b.group("Variation prompt", [we, mlat, note], "#355")
    b.group("Batched generation", [mks, mdec, nface], "#553")
    return b.build()

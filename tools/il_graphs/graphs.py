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


def build_dataset(name, identity, outfit, vary_outfit=False):
    """Synthetic training-data generator for ONE roster character (one graph per CHARACTERS entry).

    A fixed-seed HERO portrait is the only identity source: it feeds an IPAdapter PLUS-FACE
    that pins that face onto every render, while an Impact wildcard prompt varies
    (outfit) / pose / angle / framing / expression and the Gen Seed re-rolls. Curate the on-model
    outputs into a ~30-image set, then train_lora.ps1 -Char <name>. No external image loads.
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

    # --- IPAdapter PLUS-FACE: builds a hero-identity-locked model used ONLY by the face detailer
    # below (NOT the base render — putting IPAdapter on the whole gen washes/softens it). ---
    ipl = b.add("IPAdapterUnifiedLoader", ["PLUS FACE (portraits)"], pos=(-1400, 60), title="IPAdapter loader")
    # K+V (not "V only") transfers far more of the hero's identity — safe here because IPAdapter
    # touches ONLY the face crop, so the old "K+V bleaches the whole scene" problem can't happen.
    ipa = b.add("IPAdapterAdvanced", [0.85, "ease in-out", "concat", 0, 1, "K+V"],
                pos=(-1400, 230), title="IPAdapter apply (face lock, 0.85 K+V)")
    b.link(ck, "MODEL", ipl, "model")
    b.link(ipl, "model", ipa, "model"); b.link(ipl, "ipadapter", ipa, "ipadapter")
    b.link(hdec, "IMAGE", ipa, "image")

    # --- variation prompt: identity + outfit + wildcards (reroll Gen Seed to repopulate) ---
    # vary_outfit True -> __outfit__ wildcard (swappable-outfit LoRA); False -> fixed outfit (signature).
    outfit_tok = "__outfit__" if vary_outfit else outfit
    wtext = identity + ", " + outfit_tok + ", __framing__, __angle__, __pose__, __expression__"
    populated = identity + ", " + outfit + ", upper body, front view, standing, neutral expression"
    we = b.add("ImpactWildcardEncode",
               [wtext, populated, "populate", "Select the LoRA to add to the text",
                "Select the Wildcard to add to the text", SEED, "randomize"],
               pos=(-1060, -100), title="Wildcard prompt (outfit/pose/angle/framing/expr)")
    # base render uses the RAW checkpoint (clean, crisp); IPAdapter is applied only at the face pass.
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

    # --- lock the hero face IN THE FACE CROP ONLY (clean base elsewhere), pose-NEUTRAL cond ---
    nface = b.add("CLIPTextEncode", [identity], pos=(-400, 220), title="Face detail (neutral identity)")
    b.link(clip, "CLIP", nface, "clip")
    c = dict(msrc=(we, "model"), clip=clip, vae=vae,   # msrc = raw checkpoint (base + hand stay clean)
             cpos=(we, "conditioning"), cneg=(neg, "CONDITIONING"), seed=gseed)
    h = add_detailers(b, c, (mdec, "IMAGE"), x=-60, face_cond=(nface, "CONDITIONING"),
                      face_model=(ipa, "MODEL"), face_denoise=0.5)   # face pass = IPAdapter hero lock

    note = b.add("Note", [
        f"DATASET TOOL for character '{name}' (one graph per CHARACTERS entry in config).\n"
        "1. Hero Seed fixed; pick a hero portrait you like (it sets the face).\n"
        f"2. Reroll Gen Seed (batch of 4) -> ~60 varied shots into output/dataset/{name}/.\n"
        "3. Delete the off-model ones IN PLACE (curation) -- no file moving.\n"
        f"4. Train:  tools/lora_train/train_lora.ps1 -Char {name}   (auto-captions + trains)\n"
        "   or train every roster character at once:  tools/lora_train/train_all.ps1\n"
        "Edit identity/outfit/vary_outfit in CHARACTERS (config.py) + regenerate.\n"
        "Wildcards: outfit/pose/angle/framing/expression .txt in ComfyUI-Impact-Pack/wildcards/."],
        pos=(-1060, 500), title=f"How to use ({name})", color=NOTE_C, bgcolor=NOTE_BG)

    # SaveImage splits the prefix on the LAST "/": "dataset/<name>/<name>" => a real per-character
    # subfolder output/dataset/<name>/ (just "dataset/<name>" would dump every character into output/dataset/).
    add_finish(b, h, f"dataset/{name}/{name}", x=1100)
    b.group("Load + Seeds", [ck, vae, hseed, gseed, clip, neg], "#535")
    b.group("Hero portrait (identity source)", [hpos, hlat, hks, hdec, hprev], "#525")
    b.group("IPAdapter face lock", [ipl, ipa], "#525")
    b.group("Variation prompt", [we, mlat, note], "#355")
    b.group("Batched generation", [mks, mdec, nface], "#553")
    return b.build()

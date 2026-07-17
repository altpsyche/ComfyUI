from __future__ import annotations
from .builder import Builder
from .config import (ROOT, CKPT, VAE, SEED, POS, NEG, REF_SUFFIX,
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


def build_xyplot():
    """Pick-the-best-epoch tool: an XY grid of LoRA epoch (X) x strength (Y) via efficiency-nodes.

    After training, kohya saves one LoRA per epoch (`<char>_v1-000001..` + the final `<char>_v1`). This
    graph renders the SAME prompt/seed across a batch of those files crossed with a weight sweep, so you
    can eyeball which (epoch, strength) is the best likeness without frying. Generic — point it at any
    character: drop the epochs to compare into models/loras/_xyplot/ and add that character's trigger word.
    """
    b = Builder()
    # The LoRA-Batch loader requires an ABSOLUTE path: it loads each discovered file directly only when
    # os.path.isabs() is true; a relative path is sent through folder_paths.get_full_path() which returns
    # None for a nested folder -> "Error loading Lora file ... 'NoneType' has no attribute 'lower'".
    xy_dir = str(ROOT / "models" / "loras" / "_xyplot")
    loader = b.add("Efficient Loader",
                   [CKPT, VAE, -2, "None", 1.0, 1.0, POS, NEG, "none", "comfy", 832, 1216, 1],
                   pos=(0, 0), title="Efficient Loader (clip skip -2)")
    # X = a batch of LoRA files (the epochs) from xy_dir, Y = a weight sweep.
    lplot = b.add("XY Input: LoRA Plot",
                  ["X: LoRA Batch, Y: LoRA Weight", "None", 1.0, 1.0, 10, xy_dir,
                   False, "ascending", 0.0, 1.0, 3, 0.5, 0.9],
                  pos=(0, 440), title="X: epochs (folder)  |  Y: weight 0.5->0.9")
    xyp = b.add("XY Plot", [10, "False", "Horizontal", "True", "Plot"],
                pos=(460, 440), title="XY Plot")
    ks = b.add("KSampler (Efficient)",
               [SEED, "fixed", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0, "auto", "true"],
               pos=(460, 0), title="KSampler (Efficient) -> grid")
    b.link(loader, "MODEL", ks, "model")
    b.link(loader, "CONDITIONING+", ks, "positive"); b.link(loader, "CONDITIONING-", ks, "negative")
    b.link(loader, "LATENT", ks, "latent_image"); b.link(loader, "VAE", ks, "optional_vae")
    b.link(loader, "DEPENDENCIES", xyp, "dependencies")
    b.link(lplot, "X", xyp, "X"); b.link(lplot, "Y", xyp, "Y")
    b.link(xyp, "SCRIPT", ks, "script")
    b.add("Note", [
        "PICK-THE-BEST-EPOCH GRID (efficiency-nodes XY Plot). One queue = a grid of epoch x strength.\n"
        "1. Put the epochs you want to compare in  models/loras/_xyplot/  (copy e.g. ursa_v1.safetensors\n"
        "   + ursa_v1-000008.safetensors + ...). That folder is the X axis; raise X_batch_count if needed.\n"
        "2. In the Efficient Loader's positive prompt, ADD the character's trigger word (e.g. 'ursachar').\n"
        "3. Y axis = LoRA weight, swept 0.5 -> 0.9 over 3 columns (edit Y_first/Y_last/Y_batch_count).\n"
        "4. Keep the seed FIXED so every cell is the same image; Queue ONCE -> output/xyplot/.\n"
        "Pick the (epoch, weight) cell with the best likeness that isn't over-cooked / over-saturated.\n"
        "X_batch_path is an ABSOLUTE path (the LoRA-Batch loader requires it); repoint it if your\n"
        "epochs live elsewhere. Leave the Efficient Loader's lora_name = None (the grid supplies the LoRA)."],
        pos=(920, 440), title="How to use", color=NOTE_C, bgcolor=NOTE_BG)
    add_finish(b, (ks, "IMAGE"), "xyplot/epoch_grid", x=920)
    b.group("Load (clip skip -2)", [loader], "#535")
    b.group("XY: epochs x strength", [lplot, xyp], "#355")
    b.group("Sample -> grid", [ks], "#553")
    return b.build()


# Qwen-Image-Edit-2511 model stack (downloaded by `dev models install il_graphs`).
QE_GGUF = "qwen-image-edit-2511-Q5_K_M.gguf"
QE_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QE_VAE = "qwen_image_vae.safetensors"
QE_LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QE_ANGLES = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

# Stage-2 edit knobs (the tunable Qwen-Edit dials — change here, regenerate). See DATASET.md.
QE_LIGHTNING_STR = 1.0     # Lightning 4-step LoRA strength
QE_ANGLES_STR = 0.8        # multiple-angles LoRA: raise toward 1.0 for more angle push, lower if identity drifts
QE_SHIFT = 3.1             # ModelSamplingAuraFlow shift (official 2511 value)
QE_CFGNORM = 1.0           # CFGNorm
QE_EDIT_STEPS = 6          # Lightning edit steps; raise to 8-10 only if edits look soft
QE_EDIT_CFG = 1.0          # Lightning runs at cfg 1.0 (so the negative branch is unused)
QE_EDIT_SAMPLER = "euler"
QE_EDIT_SCHED = "simple"
# Hero detail (Stage 1b): face+hand detail the ONE hero in YOUR SDXL checkpoint BEFORE it feeds Qwen, so a
# crisp, on-model character propagates into every edited frame. This is ONE detail pass total (the hero),
# vs the old per-frame Stage-3 pass (one per saved image) -> much faster, and the quality is set at the
# source. The face pass uses a clean identity-only prompt (outfit/pose tags fight the tiny face crop).
QE_HERO_DETAIL = True
QE_HERO_FACE_DENOISE = 0.35     # hero face re-render strength (higher = more SDXL face assertion)
# Stage-3 polish: optionally ALSO re-detail every edited frame (belt-and-suspenders if Qwen still softens
# faces). Off by default now that the hero is detailed up front; set True to re-enable the per-frame pass.
QE_STAGE3_POLISH = False
QE_STAGE3_FACE_DENOISE = 0.35   # face re-render strength (higher = more re-assertion of the SDXL face)


def build_dataset_edit(name="edit", identity="1girl, solo", outfit="", hero_seed=SEED, framing=None,
                       *, save_tag=None, train_char=None):
    """FRONTIER dataset generator (self-contained): generate ONE original hero, then re-pose it into
    many varied shots with Qwen-Image-Edit-2511 (GGUF), holding identity + art style. One graph per
    roster character (IL_DatasetEdit_<name>); the trainer reads output/dataset/<name>/ as usual.

    Stage 1 (bootstrap the hero): an Illustrious text2img renders the character from its `id` (+ `outfit`)
    tags at a fixed Hero Seed -> HERO preview. Reroll the Hero Seed to pick a face you like; that single
    image is the identity anchor. Stage 2 (propagate): FluxKontextImageScale -> TextEncodeQwenImageEditPlus
    (scaled hero + a wildcard instruction) -> KSampler on the GGUF model (Lightning 4-step, 6 steps /
    cfg 1.0) -> save to output/dataset/<name>/. Because the hero is rendered in YOUR checkpoint, the
    edits stay on-style; the edit only changes pose/angle/expression. Verified on the live ComfyUI.

    MODULAR character (one graph per outfit): the outfit's garments go in the HERO prompt (SDXL renders
    the character wearing it -- highest fidelity, the proven path; Qwen then only re-poses, preserving the
    outfit). Faces across outfits are "recognizably the same person" (same hero_seed + id), which is all
    the dataset needs -- the always-on identity token averages them into one face at train time, while a
    Qwen re-dress would instead poison the outfit token with a low-fidelity edit. `save_tag`
    (e.g. "mira/winter") sets the output subfolder output/dataset/<save_tag>/; `train_char` is the
    character the how-to note tells you to train (the base char, not the per-outfit name).

    Layout: clean left->right columns (Stage1 hero | model | encoders | instruction | refs | edit).
    """
    save_tag = save_tag or name
    train_char = train_char or name
    b = Builder()
    # ===== STAGE 1 — Illustrious hero generator (text2img from the character's id; pick a face) =====
    ck = b.add("CheckpointLoaderSimple", [CKPT], pos=(0, 0), title="Checkpoint (Illustrious)")
    hvae = b.add("VAELoader", [VAE], pos=(0, 160), title="VAE (hero)")
    hseed = b.add("Seed", [hero_seed, "fixed"], pos=(0, 300), title="Hero Seed (fixed = pick the face)")
    hclip = b.add("CLIPSetLastLayer", [-2], pos=(0, 450), title="CLIP skip -2")
    b.link(ck, "CLIP", hclip, "clip")
    # Hero framing suffix: per-character `framing` (full descriptor, no leading comma) overrides the
    # global REF_SUFFIX — e.g. a face-focused character can use a portrait crop for a larger face.
    suffix = (", " + framing) if framing else REF_SUFFIX
    # The outfit (signature, or a modular character's per-graph outfit) is rendered into the hero by SDXL,
    # so it's high-fidelity and Qwen only has to preserve it while re-posing.
    hpos = b.add("CLIPTextEncode", [identity + (", " + outfit if outfit else "") + suffix],
                 pos=(360, 0), title="Hero prompt (identity + outfit)")
    hneg = b.add("CLIPTextEncode", [NEG], pos=(360, 180), title="Negative")
    hlat = b.add("EmptyLatentImage", [832, 1216, 1], pos=(360, 360), title="Hero latent 832x1216")
    b.link(hclip, "CLIP", hpos, "clip"); b.link(hclip, "CLIP", hneg, "clip")
    hks = b.add("KSampler", [hero_seed, "fixed", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0],
                pos=(720, 0), title="Hero KSampler")
    b.link(ck, "MODEL", hks, "model"); b.link(hpos, "CONDITIONING", hks, "positive")
    b.link(hneg, "CONDITIONING", hks, "negative"); b.link(hlat, "LATENT", hks, "latent_image")
    b.link(hseed, "int", hks, "seed")
    hdec = b.add("VAEDecode", [], pos=(720, 320), title="Hero decode")
    b.link(hks, "LATENT", hdec, "samples"); b.link(hvae, "VAE", hdec, "vae")

    # ===== STAGE 1b — detail the HERO once (face + hand) in YOUR SDXL checkpoint, BEFORE Qwen =====
    # A crisp, on-model hero propagates into every edited frame -> high quality at the source, and only
    # ONE detail pass total (vs the old per-frame Stage-3 pass). The face crop uses an identity-only
    # prompt (outfit/pose/framing tags fight the tiny face crop -- see GOTCHAS).
    hero_img = (hdec, "IMAGE")
    if QE_HERO_DETAIL:
        hfpos = b.add("CLIPTextEncode", [identity], pos=(360, 640), title="Hero face prompt (identity only)")
        b.link(hclip, "CLIP", hfpos, "clip")
        chero = {"msrc": (ck, "MODEL"), "clip": hclip, "vae": hvae, "seed": hseed,
                 "cpos": (hpos, "CONDITIONING"), "cneg": (hneg, "CONDITIONING")}
        hero_img = add_detailers(b, chero, (hdec, "IMAGE"), x=820,
                                 face_cond=(hfpos, "CONDITIONING"), face_denoise=QE_HERO_FACE_DENOISE)
    hprev = b.add("PreviewImage", [], pos=(720, 470), title="HERO preview (reroll Hero Seed to pick the face)")
    b.link(hero_img[0], hero_img[1], hprev, "images")   # preview the DETAILED hero (what feeds Qwen)

    # ===== STAGE 2 model — GGUF + Lightning(+angles) LoRA -> flow-shift + CFGNorm (2511 patch chain) =====
    gguf = b.add("UnetLoaderGGUF", [QE_GGUF], pos=(2180, 0), title="Qwen-Edit GGUF (Q5)")
    llora = b.add("LoraLoaderModelOnly", [QE_LIGHTNING, QE_LIGHTNING_STR], pos=(2180, 150), title="Lightning 4-step LoRA")
    alora = b.add("LoraLoaderModelOnly", [QE_ANGLES, QE_ANGLES_STR], pos=(2180, 300), title="Multiple-angles LoRA")
    msaf = b.add("ModelSamplingAuraFlow", [QE_SHIFT], pos=(2180, 450), title="ModelSampling (shift 3.1)")
    cfgn = b.add("CFGNorm", [QE_CFGNORM], pos=(2180, 600), title="CFGNorm")
    b.link(gguf, "MODEL", llora, "model"); b.link(llora, "MODEL", alora, "model")
    b.link(alora, "MODEL", msaf, "model"); b.link(msaf, "MODEL", cfgn, "model")

    # ===== encoders + scale =====
    clip = b.add("CLIPLoader", [QE_CLIP, "qwen_image", "default"], pos=(2620, 0), title="Qwen 2.5-VL text encoder")
    vae = b.add("VAELoader", [QE_VAE], pos=(2620, 170), title="Qwen Image VAE")
    scale = b.add("FluxKontextImageScale", [], pos=(2620, 340), title="Scale ref (hero -> edit)")
    b.link(hero_img[0], hero_img[1], scale, "image")   # the DETAILED Stage-1 hero feeds the edit

    # ===== instruction + encode (wildcards vary angle/pose/expr + new framing/background/lighting) =====
    # Keep the identity-preamble comma-list that empirically gave good POSE/ANGLE variety, and just
    # APPEND the new framing/background/lighting axes. (A "lead with the imperative change" rewrite was
    # tried and REVERTED: leading with framing buried __pose__/__angle__ behind the easy global edits,
    # so Qwen -- conservative at 6 steps -- moved the pose less. Don't reorder pose behind framing/scene.)
    # mode "populate": the UI re-expands wildcard_text into the populated_text box each queue, so you SEE
    # the resolved prompt and it re-rolls. populated_text also holds the wildcard string (not a concrete
    # roll) so a headless API POST -- no frontend -- still expands them in the node BACKEND
    # (process(populated_text, seed)); that is the only headless-correctness change kept from the rewrite.
    wtext = ("same character, identical face and hair and outfit, keep the same art style, "
             "__angle__, __pose__, __expression__, __framing__, __background__, __lighting__")
    wild = b.add("ImpactWildcardProcessor",
                 [wtext, wtext, "populate", SEED, "randomize", "Select the Wildcard to add to the text"],
                 pos=(3120, 0), title="Edit instruction (reroll = variety)")
    posenc = b.add("TextEncodeQwenImageEditPlus", [""], pos=(3120, 220), title="Encode (positive: hero + instruction)")
    negenc = b.add("TextEncodeQwenImageEditPlus", [""], pos=(3120, 470), title="Encode (negative: empty)")
    for enc in (posenc, negenc):
        b.link(clip, "CLIP", enc, "clip"); b.link(vae, "VAE", enc, "vae"); b.link(scale, "IMAGE", enc, "image1")
    b.link(wild, "STRING", posenc, "prompt")   # instruction drives the positive encoder

    # ===== reference-latent method (needed for repackaged/GGUF builds) + init latent =====
    posref = b.add("FluxKontextMultiReferenceLatentMethod", ["index_timestep_zero"], pos=(3660, 220), title="Ref method (pos)")
    negref = b.add("FluxKontextMultiReferenceLatentMethod", ["index_timestep_zero"], pos=(3660, 470), title="Ref method (neg)")
    b.link(posenc, "CONDITIONING", posref, "conditioning"); b.link(negenc, "CONDITIONING", negref, "conditioning")
    venc = b.add("VAEEncode", [], pos=(3660, 640), title="Encode hero -> latent")
    b.link(scale, "IMAGE", venc, "pixels"); b.link(vae, "VAE", venc, "vae")

    # ===== edit + decode (Lightning: 6 steps / cfg 1.0 / euler / simple) =====
    ks = b.add("KSampler", [SEED, "randomize", QE_EDIT_STEPS, QE_EDIT_CFG, QE_EDIT_SAMPLER, QE_EDIT_SCHED, 1.0],
               pos=(4120, 220), title="Edit KSampler (6 steps)")
    b.link(cfgn, "MODEL", ks, "model"); b.link(posref, "CONDITIONING", ks, "positive")
    b.link(negref, "CONDITIONING", ks, "negative"); b.link(venc, "LATENT", ks, "latent_image")
    vdec = b.add("VAEDecode", [], pos=(4120, 460), title="Decode")
    b.link(ks, "LATENT", vdec, "samples"); b.link(vae, "VAE", vdec, "vae")

    note = b.add("Note", [
        f"QWEN-EDIT DATASET TOOL ('{name}') -- self-contained: it MAKES the hero, then re-poses it.\n"
        "STAGE 1 (left): reroll 'Hero Seed' and watch HERO preview until you like the face (rendered\n"
        "from this character's id tags in YOUR checkpoint, then face+hand DETAILED so it's crisp). Then\n"
        "leave Hero Seed fixed on that value. (Reroll feels slow? mute the 'Face + Hand Detail' group.)\n"
        "STAGE 2: leave 'Edit instruction' seed control = randomize; set batch count ~40 + Queue once\n"
        "  -> output/dataset/" + save_tag + "/. Each frame = that DETAILED hero re-posed into a new framing/\n"
        "  angle/pose/expression/background/lighting, same identity + art style. (mode 'populate': the\n"
        "  bottom box shows the resolved prompt and re-rolls each queue; also expands headless via backend.)\n"
        "Then curate the best 25-40 and run: ./dev train " + train_char + ".\n"
        "Wildcards (__angle__/__pose__/__expression__/__framing__/__background__/__lighting__) live in\n"
        "  ComfyUI-Impact-Pack/wildcards/. Too slow / OOM? re-run ./dev models install il_graphs --variant quant=Q4_K_M.\n"
        "Poses too similar? __pose__/__angle__ lead the instruction on purpose -- raise the multiple-angles\n"
        "  LoRA toward 1.0, or drop a scene axis. Identity drifting? lower the multiple-angles LoRA.\n"
        "QUALITY: the HERO is detailed up front (one pass) so every frame inherits a crisp face. If Qwen\n"
        "  still softens faces, set QE_STAGE3_POLISH=True in il_graphs/graphs.py to ALSO detail each frame."],
        pos=(3120, 700), title=f"How to use ({name})", color=NOTE_C, bgcolor=NOTE_BG)

    # ===== STAGE 3 (optional) — ALSO re-detail each edited frame in YOUR SDXL checkpoint (face + hand) =====
    # Off by default: the hero is already detailed up front (Stage 1b), so the face is crisp at the source.
    # Enable (QE_STAGE3_POLISH) for a belt-and-suspenders per-frame pass if Qwen still softens faces. The
    # face pass uses an identity-ONLY prompt (outfit/pose/framing tags fight the tiny face crop).
    final = (vdec, "IMAGE")
    if QE_STAGE3_POLISH:
        fpos = b.add("CLIPTextEncode", [identity], pos=(4120, 640), title="Face restore prompt (identity only)")
        b.link(hclip, "CLIP", fpos, "clip")
        cpolish = {"msrc": (ck, "MODEL"), "clip": hclip, "vae": hvae, "seed": hseed,
                   "cpos": (hpos, "CONDITIONING"), "cneg": (hneg, "CONDITIONING")}
        final = add_detailers(b, cpolish, (vdec, "IMAGE"), x=4600,
                              face_cond=(fpos, "CONDITIONING"), face_denoise=QE_STAGE3_FACE_DENOISE)

    add_finish(b, final, f"dataset/{save_tag}/{save_tag.split('/')[-1]}", x=5520)
    stage1 = [ck, hvae, hseed, hclip, hpos, hneg, hlat, hks, hdec, hprev]
    if QE_HERO_DETAIL:
        stage1.append(hfpos)
    b.group("STAGE 1 - Hero generator + detail (Illustrious)", stage1, "#535")
    b.group("STAGE 2 - Qwen-Edit model + LoRAs", [gguf, llora, alora, msaf, cfgn], "#525")
    b.group("Encoders + scale", [clip, vae, scale], "#535")
    b.group("Instruction + encode", [wild, posenc, negenc, note], "#355")
    b.group("Reference + latent", [posref, negref, venc], "#355")
    b.group("Edit + decode", [ks, vdec], "#553")
    return b.build()

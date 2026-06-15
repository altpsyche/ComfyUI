from __future__ import annotations
import copy
from .config import (SEED, CKPT, VAE, UPSCALE, CN_UNION,
                     BASE_SAMPLER, BASE_SCHED, BASE_STEPS, BASE_CFG,
                     POS, NEG, HAND_POS, FACE_POS, NOTE_C, NOTE_BG)
from .templates import TEMPLATES


def lora_wv(entries):
    body = [{"on": on, "lora": f, "strength": s, "strengthTwo": None} for (f, on, s) in entries]
    return [{}, {"type": "PowerLoraLoaderHeaderWidget"}, *body, {}, ""]


def core(b, ipadapter=False, pose=False, lcm=False, loras=None):
    """Build the base txt2img block.

    With no feature flags the output is IDENTICAL across tiers (raw checkpoint,
    euler_ancestral/normal/30/cfg5, fixed seed) — that is the comparison ladder.
    Every graph carries a LoRA bank, but it is a pass-through until a LoRA is toggled
    on, so the default output is unchanged.
    The feature flags build *active* (not bypassed) feature graphs:
      ipadapter=True  active IPAdapter (plus-face) with a ref-image slot
      pose=True       active OpenPose ControlNet on the base conditioning
      lcm=True        pre-load lcm-lora ON + LCM sampler config (8 steps, cfg 1.5)
      loras=[(file, on, strength), ...]  pre-populate the LoRA bank (e.g. a character LoRA)
    Returns node ids + (node,slot) model/conditioning sources for the layer helpers.
    """
    c = {}
    ck = b.add("CheckpointLoaderSimple", [CKPT], pos=(-2700, -100), title="Checkpoint")
    vae = b.add("VAELoader", [VAE], pos=(-2700, 60), title="VAE")
    seed = b.add("Seed", [SEED, "fixed"], pos=(-2700, 220), title="Seed (fixed)")
    c.update(ck=ck, vae=vae, seed=seed)

    # LoRA bank — always present, opt-in. model/clip route THROUGH it; every downstream layer
    # (upscale, detailers, inpaint, bg) reads c["msrc"], so a LoRA toggled on here applies to
    # the WHOLE pipeline. All entries OFF = pass-through (ladder stays identical). lcm just
    # pre-loads lcm-lora ON.
    msrc, csrc, feat_ids = (ck, "MODEL"), (ck, "CLIP"), []
    entries = list(loras or [])
    if lcm:
        entries.append(("lcm-lora-sdxl.safetensors", True, 1.0))
    if not entries:
        entries = [("put_character_lora_here.safetensors", False, 1.0)]
    bank = b.add("Power Lora Loader (rgthree)", lora_wv(entries),
                 pos=(-2400, -100), title="LoRA bank")
    b.link(ck, "MODEL", bank, "model"); b.link(ck, "CLIP", bank, "clip")
    msrc, csrc = (bank, "MODEL"), (bank, "CLIP"); feat_ids.append(bank)
    if ipadapter:
        ipl = b.add("IPAdapterUnifiedLoader", ["PLUS FACE (portraits)"], pos=(-2400, 300),
                    title="IPAdapter loader")
        ipa = b.add("IPAdapterAdvanced", [0.7, "ease in-out", "concat", 0, 1, "V only"],
                    pos=(-2400, 470), title="IPAdapter apply (weight 0.7)")
        ipimg = b.add("LoadImage", ["example.png", "image"], pos=(-2700, 470), title="IPAdapter ref >> LOAD")
        b.link(*msrc, ipl, "model")
        b.link(ipl, "model", ipa, "model"); b.link(ipl, "ipadapter", ipa, "ipadapter")
        b.link(ipimg, "IMAGE", ipa, "image")
        msrc = (ipa, "MODEL"); feat_ids += [ipl, ipa, ipimg]

    clip = b.add("CLIPSetLastLayer", [-2], pos=(-2060, -100), title="CLIP skip -2")
    b.link(*csrc, clip, "clip")
    pos = b.add("CLIPTextEncode", [POS], pos=(-1780, -200), title="Positive")
    neg = b.add("CLIPTextEncode", [NEG], pos=(-1780, 60), title="Negative")
    b.link(clip, "CLIP", pos, "clip"); b.link(clip, "CLIP", neg, "clip")
    lat = b.add("EmptyLatentImage", [832, 1216, 1], pos=(-1780, 320), title="Empty Latent 832x1216")

    cpos, cneg, pose_ids = (pos, "CONDITIONING"), (neg, "CONDITIONING"), []
    if pose:
        pimg = b.add("LoadImage", ["example.png", "image"], pos=(-1780, -640), title="Pose ref >> LOAD")
        dwp = b.add("DWPreprocessor",
                    ["enable", "enable", "enable", 512, "yolox_l.onnx", "dw-ll_ucoco_384_bs5.torchscript.pt", "disable"],
                    pos=(-1440, -640), title="DWPose")
        ped = b.add("OpenposeEditorNode", copy.deepcopy(TEMPLATES["OpenposeEditorNode"]["widgets_values"]),
                    pos=(-1100, -640), title="Pose edit")
        pcn = b.add("ControlNetLoader", [CN_UNION], pos=(-1780, -380), title="ControlNet union (pose)")
        ptype = b.add("SetUnionControlNetType", ["openpose"], pos=(-760, -640), title="Union type: openpose")
        papply = b.add("ControlNetApplyAdvanced", [0.7, 0, 0.8], pos=(-760, -380), title="Apply pose (str 0.7)")
        b.link(pimg, "IMAGE", dwp, "image"); b.link(dwp, "POSE_KEYPOINT", ped, "POSE_KEYPOINT")
        b.link(pcn, "CONTROL_NET", ptype, "control_net"); b.link(ptype, "CONTROL_NET", papply, "control_net")
        b.link(ped, "POSE_IMAGE", papply, "image")
        b.link(pos, "CONDITIONING", papply, "positive"); b.link(neg, "CONDITIONING", papply, "negative")
        cpos, cneg = (papply, "positive"), (papply, "negative")
        pose_ids = [pimg, dwp, ped, pcn, ptype, papply]

    if lcm:
        ks_wv = [SEED, "fixed", 8, 1.5, "lcm", "sgm_uniform", 1.0]
    else:
        ks_wv = [SEED, "fixed", BASE_STEPS, BASE_CFG, BASE_SAMPLER, BASE_SCHED, 1.0]
    ks = b.add("KSampler", ks_wv, pos=(-1440, -100), title="KSampler (base)")
    b.link(*msrc, ks, "model")
    b.link(*cpos, ks, "positive"); b.link(*cneg, ks, "negative")
    b.link(lat, "LATENT", ks, "latent_image"); b.link(seed, "int", ks, "seed")
    dec = b.add("VAEDecode", [], pos=(-1100, -100), title="VAE Decode (base)")
    b.link(ks, "LATENT", dec, "samples"); b.link(vae, "VAE", dec, "vae")

    c.update(clip=clip, pos=pos, neg=neg, lat=lat, ks=ks, dec=dec,
             msrc=msrc, cpos=cpos, cneg=cneg)
    b.group("Load + Seed", [ck, vae, seed], "#535")
    if feat_ids:
        b.group("LoRA bank + IPAdapter" if ipadapter else "LoRA bank", feat_ids, "#525")
    b.group("Prompt", [pos, neg, lat], "#355")
    if pose_ids:
        b.group("OpenPose control", pose_ids, "#933")
    b.group("Base Gen", [ks, dec], "#553")
    return c


def add_upscale(b, c, with_cn=False, x=-760):
    upm = b.add("UpscaleModelLoader", [UPSCALE], pos=(x, 280), title="Upscale model")
    usdu = b.add("UltimateSDUpscale",
                 [1.5, SEED, "fixed", 22, BASE_CFG, "euler_ancestral", "karras", 0.20, "Linear",
                  1024, 1024, 8, 96, "Half Tile", 1, 64, 8, 16, True, False, 1],
                 pos=(x + 620, -100), title="Ultimate SD Upscale 1.5x")
    upos, uneg, cn_ids = c["cpos"], c["cneg"], []
    if with_cn:
        cnl = b.add("ControlNetLoader", [CN_UNION], pos=(x, 460), title="ControlNet union (up)")
        soft = b.add("ACN_ScaledSoftControlNetWeights", [0.825, 1], pos=(x, 620), title="CN soft weights")
        dpre = b.add("DepthAnythingPreprocessor", ["depth_anything_vitl14.pth", 1248], pos=(x + 300, 460), title="Depth pre")
        lpre = b.add("LineArtPreprocessor", ["enable", 1248], pos=(x + 300, 600), title="LineArt pre")
        dt = b.add("SetUnionControlNetType", ["depth"], pos=(x + 300, 280), title="type: depth")
        lt = b.add("SetUnionControlNetType", ["canny/lineart/anime_lineart/mlsd"], pos=(x + 300, 760), title="type: lineart")
        dap = b.add("ACN_AdvancedControlNetApply_v2", [0.35, 0, 0.7], pos=(x + 560, 360), title="Apply depth")
        lap = b.add("ACN_AdvancedControlNetApply_v2", [0.25, 0, 0.6], pos=(x + 560, 620), title="Apply lineart")
        b.link(c["dec"], "IMAGE", dpre, "image"); b.link(c["dec"], "IMAGE", lpre, "image")
        b.link(cnl, "CONTROL_NET", dt, "control_net"); b.link(cnl, "CONTROL_NET", lt, "control_net")
        b.link(*c["cpos"], dap, "positive"); b.link(*c["cneg"], dap, "negative")
        b.link(dt, "CONTROL_NET", dap, "control_net"); b.link(dpre, "IMAGE", dap, "image")
        b.link(soft, "CN_WEIGHTS", dap, "weights_override")
        b.link(dap, "positive", lap, "positive"); b.link(dap, "negative", lap, "negative")
        b.link(lt, "CONTROL_NET", lap, "control_net"); b.link(lpre, "IMAGE", lap, "image")
        b.link(soft, "CN_WEIGHTS", lap, "weights_override")
        upos, uneg = (lap, "positive"), (lap, "negative")
        cn_ids = [cnl, soft, dpre, lpre, dt, lt, dap, lap]
        b.group("ControlNet @ upscale", cn_ids, "#363")
    b.link(c["dec"], "IMAGE", usdu, "image"); b.link(*c["msrc"], usdu, "model")
    b.link(*upos, usdu, "positive"); b.link(*uneg, usdu, "negative")
    b.link(c["vae"], "VAE", usdu, "vae"); b.link(upm, "UPSCALE_MODEL", usdu, "upscale_model")
    b.link(c["seed"], "int", usdu, "seed")
    b.group("Hires Upscale", [upm, usdu], "#357")
    return (usdu, "IMAGE")


def _fd(b, widgets_pos, title, denoise):
    wv = [512, True, 1024, SEED, "fixed", 20, BASE_CFG, "euler_ancestral", "karras", denoise,
          5, True, True, 0.3, 10, 3, "center-1", 0, 0.93, 0, 0.7, "False", 10, "", 1, False, 40, False, False]
    return b.add("FaceDetailer", wv, pos=widgets_pos, title=title)


def add_detailers(b, c, image_src, x=540, face_cond=None, face_model=None, face_denoise=0.3):
    # face_model lets the FACE pass run on a different model than the base/hand (e.g. an IPAdapter
    # identity-locked model) so the rest of the image keeps the clean base render. Defaults =
    # c["msrc"] / 0.3 → identical to the plain tiers.
    fdet = b.add("UltralyticsDetectorProvider", ["bbox/face_yolov9c.pt"], pos=(x, 360), title="Face detector")
    hdet = b.add("UltralyticsDetectorProvider", ["bbox/hand_yolov9c.pt"], pos=(x, 500), title="Hand detector")
    sam = b.add("SAMLoader", ["sam2_hiera_large.pt", "AUTO"], pos=(x, 640), title="SAM2")
    hpos = b.add("CLIPTextEncode", [HAND_POS], pos=(x, 760), title="Hand positive")
    face = _fd(b, (x + 320, -100), "Face Detailer", face_denoise)
    hand = _fd(b, (x + 740, -100), "Hand Detailer", 0.3)
    fcond = face_cond or c["cpos"]   # neutral face prompt for multi-char; combined cond otherwise
    b.link(image_src[0], image_src[1], face, "image"); b.link(*(face_model or c["msrc"]), face, "model")
    b.link(c["clip"], "CLIP", face, "clip"); b.link(c["vae"], "VAE", face, "vae")
    b.link(*fcond, face, "positive"); b.link(*c["cneg"], face, "negative")
    b.link(fdet, "BBOX_DETECTOR", face, "bbox_detector"); b.link(sam, "SAM_MODEL", face, "sam_model_opt")
    b.link(c["seed"], "int", face, "seed")
    b.link(c["clip"], "CLIP", hpos, "clip")
    b.link(face, "image", hand, "image"); b.link(*c["msrc"], hand, "model")
    b.link(c["clip"], "CLIP", hand, "clip"); b.link(c["vae"], "VAE", hand, "vae")
    b.link(hpos, "CONDITIONING", hand, "positive"); b.link(*c["cneg"], hand, "negative")
    b.link(hdet, "BBOX_DETECTOR", hand, "bbox_detector"); b.link(sam, "SAM_MODEL", hand, "sam_model_opt")
    b.link(c["seed"], "int", hand, "seed")
    b.group("Face + Hand Detail", [fdet, hdet, sam, hpos, face, hand], "#735")
    return (hand, "image")


def add_face_inpaint(b, c, image_src, x=1700):
    fdet = b.add("UltralyticsDetectorProvider", ["bbox/face_yolov9c.pt"], pos=(x, 600), title="Face detector (inpaint)")
    fpos = b.add("CLIPTextEncode", [FACE_POS], pos=(x, 740), title="Face positive")
    fseg = b.add("BboxDetectorSEGS", [0.3, 10, 1.5, 10, "all"], pos=(x, -100), title="Face bbox SEGS")
    fmask = b.add("SegsToCombinedMask", [], pos=(x, 180), title="SEGS->mask")
    crop = b.add("InpaintCropImproved",
                 ["bilinear", "bicubic", False, "ensure minimum resolution", 1024, 1024, 16384, 16384,
                  True, 0, False, 32, 0.1, False, 1, 1, 1, 1, 1.5, True, 1024, 1024, "32", "gpu (much faster)"],
                 pos=(x, 320), title="Inpaint crop 1024")
    fenc = b.add("VAEEncode", [], pos=(x + 320, 320), title="VAE Encode (face)")
    fks = b.add("KSampler", [SEED, "fixed", 24, BASE_CFG, "euler_ancestral", "karras", 0.45],
                pos=(x + 320, 480), title="Face KSampler 0.45")
    fdc = b.add("VAEDecode", [], pos=(x + 640, 480), title="VAE Decode (face)")
    stitch = b.add("InpaintStitchImproved", [], pos=(x + 640, 320), title="Inpaint stitch")
    b.link(fdet, "BBOX_DETECTOR", fseg, "bbox_detector"); b.link(image_src[0], image_src[1], fseg, "image")
    b.link(fseg, "SEGS", fmask, "segs")
    b.link(image_src[0], image_src[1], crop, "image"); b.link(fmask, "MASK", crop, "mask")
    b.link(crop, "cropped_image", fenc, "pixels"); b.link(c["vae"], "VAE", fenc, "vae")
    b.link(*c["msrc"], fks, "model"); b.link(fpos, "CONDITIONING", fks, "positive"); b.link(*c["cneg"], fks, "negative")
    b.link(c["clip"], "CLIP", fpos, "clip")
    b.link(fenc, "LATENT", fks, "latent_image"); b.link(c["seed"], "int", fks, "seed")
    b.link(fks, "LATENT", fdc, "samples"); b.link(c["vae"], "VAE", fdc, "vae")
    b.link(crop, "stitcher", stitch, "stitcher"); b.link(fdc, "IMAGE", stitch, "inpainted_image")
    b.group("Face Inpaint (native 1024)", [fdet, fpos, fseg, fmask, crop, fenc, fks, fdc, stitch], "#737")
    return (stitch, "image")


def add_bg(b, c, image_src, x=2720):
    pdet = b.add("UltralyticsDetectorProvider", ["segm/person_yolov8s-seg.pt"], pos=(x, 760), title="Person detector")
    bmask = b.add("SegmDetectorCombined_v2", [0.5, 0], pos=(x, -100), title="Person seg mask")
    grow = b.add("GrowMask", [10, True], pos=(x, 80), title="Grow mask")
    inv = b.add("InvertMask", [], pos=(x, 240), title="Invert (bg)")
    m2s = b.add("MaskToSEGS", [False, 3, False, 10, True], pos=(x, 360), title="Mask->SEGS")
    bpipe = b.add("ToBasicPipe", [], pos=(x, 560), title="Basic pipe")
    seg = b.add("SEGSDetailer", [512, True, 1024, SEED, "fixed", 20, BASE_CFG, "euler_ancestral", "karras", 0.20,
                                 True, True, 0.2, 1, 1, False, 20], pos=(x + 320, -100), title="BG SEGS Detailer")
    paste = b.add("SEGSPaste", [10, 255], pos=(x + 320, 560), title="SEGS Paste")
    b.link(pdet, "SEGM_DETECTOR", bmask, "segm_detector"); b.link(image_src[0], image_src[1], bmask, "image")
    b.link(bmask, "MASK", grow, "mask"); b.link(grow, "MASK", inv, "mask"); b.link(inv, "MASK", m2s, "mask")
    b.link(*c["msrc"], bpipe, "model"); b.link(c["clip"], "CLIP", bpipe, "clip"); b.link(c["vae"], "VAE", bpipe, "vae")
    b.link(*c["cpos"], bpipe, "positive"); b.link(*c["cneg"], bpipe, "negative")
    b.link(image_src[0], image_src[1], seg, "image"); b.link(m2s, "SEGS", seg, "segs"); b.link(bpipe, "basic_pipe", seg, "basic_pipe")
    b.link(c["seed"], "int", seg, "seed")
    b.link(image_src[0], image_src[1], paste, "image"); b.link(seg, "segs", paste, "segs")
    b.group("Background Detail", [pdet, bmask, grow, inv, m2s, bpipe, seg, paste], "#673")
    return (paste, "IMAGE")


def add_finish(b, image_src, tag, x=3600, sharpen=False, metadata=False, aesthetic=False, lcm_note=False):
    src, ids = image_src, []
    if sharpen:
        sh = b.add("Image Lucy Sharpen", [1, 3], pos=(x, -100), title="Sharpen"); ids.append(sh)
        b.link(image_src[0], image_src[1], sh, "images"); src = (sh, "IMAGE"); x += 320
    sav = b.add("SaveImage", [tag], pos=(x, -100), title="Save"); ids.append(sav)
    b.link(src[0], src[1], sav, "images")
    if metadata:
        isave = b.add("Image Save",
                      ["[time(%Y-%m-%d)]", tag, "_", 4, "false", "png", 300, 100, "true", "false", "false",
                       "false", "true", "true", "true"], pos=(x, 160), title="Save (metadata)")
        b.link(src[0], src[1], isave, "images"); ids.append(isave)
    if aesthetic:
        a = b.add("AestheticsPredictorV2_5Node", [], pos=(x, 800), title="Aesthetic score")
        b.link(src[0], src[1], a, "image"); ids.append(a)
    if lcm_note:
        ids.append(b.add("Note", [
            "LCM fast-preview (lcm-lora wired OFF in LoRA bank):\n"
            "  1. LoRA bank -> ON 'lcm-lora-sdxl' (strength 1.0)\n"
            "  2. KSampler (base) -> sampler 'lcm', scheduler 'sgm_uniform'\n"
            "  3. steps 8, cfg 1.5\nTurn off for final renders."],
            pos=(x, 320), title="LCM fast mode", color="#583", bgcolor=NOTE_BG))
    b.group("Finish + Save", ids, "#555")

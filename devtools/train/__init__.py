"""`dev train` — train ONE character LoRA (or `--all`) with kohya sd-scripts.

Port of tools/lora_train/train_lora.ps1 + train_all.ps1. The orchestration runs on the dispatcher's
python; the venv-specific steps (WD14 tagger, accelerate) shell out to the TRAINER venv. Training
params are data-driven (train.toml, resolved by train_config.py); the dataset folder convention is
output/dataset/<char>/ (modular characters use per-outfit subfolders). Pure command-building lives
in cmd.py so it can be unit-tested.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys
from pathlib import Path

from ..core import config
from ..core import platform as plat
from ..core import venv
from . import cmd

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


class TrainError(Exception):
    pass


def _die(msg) -> None:
    raise TrainError(msg)


# flag dest -> train.toml key (only explicitly-set flags become overrides, so unset = use train.toml)
_FLAG_MAP = {
    "dim": "dim", "alpha": "alpha", "optimizer": "optimizer", "d_coef": "d_coef",
    "steps": "steps", "epochs": "epochs", "batch": "batch", "min_images": "min_images",
    "lr": "lr", "unet_lr": "unet_lr", "text_encoder_lr": "text_encoder_lr",
    "scheduler": "lr_scheduler", "min_snr": "min_snr_gamma", "save_precision": "save_precision",
    "resolution": "resolution", "save_every_n_epochs": "save_every_n_epochs",
    "bucket_min": "min_bucket_reso", "bucket_max": "max_bucket_reso",
}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dev train", description="Train a character LoRA.")
    ap.add_argument("char", nargs="?", help="dataset folder name + default LoRA name")
    ap.add_argument("--char", dest="char_opt", help=argparse.SUPPRESS)  # allow --char too
    ap.add_argument("--all", action="store_true", help="train every roster character with a dataset")
    ap.add_argument("--pack", default="il_graphs", help="pack whose [train] profile + roster to use")

    ap.add_argument("--trigger")
    ap.add_argument("--outfit")
    ap.add_argument("--prune")
    ap.add_argument("--keep")
    ap.add_argument("--base")
    ap.add_argument("--train-text-encoder", action="store_true", dest="train_text_encoder")
    ap.add_argument("--skip-caption", action="store_true", dest="skip_caption")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--profile")
    ap.add_argument("--dim", type=int)
    ap.add_argument("--alpha", type=int)
    ap.add_argument("--optimizer", choices=["prodigy", "adamw", "adafactor"])
    ap.add_argument("--d-coef", type=float, dest="d_coef")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--min-images", type=int, dest="min_images")
    ap.add_argument("--lr")
    ap.add_argument("--unet-lr", dest="unet_lr")
    ap.add_argument("--text-encoder-lr", dest="text_encoder_lr")
    ap.add_argument("--scheduler")
    ap.add_argument("--min-snr", dest="min_snr")
    ap.add_argument("--save-precision", dest="save_precision")
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--save-every-n-epochs", type=int, dest="save_every_n_epochs")
    ap.add_argument("--bucket-min", type=int, dest="bucket_min")
    ap.add_argument("--bucket-max", type=int, dest="bucket_max")
    return ap


def _lora_train_dir() -> Path:
    lt = config.TOOLS / "lora_train"
    if str(lt) not in sys.path:
        sys.path.insert(0, str(lt))
    return lt


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def _resolve_cfg(char, args):
    _lora_train_dir()
    import train_config
    overrides = {}
    for dest, key in _FLAG_MAP.items():
        val = getattr(args, dest, None)
        if val is not None:
            overrides[key] = str(val)
    if args.train_text_encoder:
        overrides["train_text_encoder"] = "true"
    cfg = train_config.resolve(char, args.profile, overrides)
    cfg["__char__"] = char
    return cfg


def _pack_train_profile(pack_name):
    from ..packs import _get
    pack = _get(pack_name)
    if pack is None:
        _die(f"unknown pack {pack_name!r}")
    prof = pack.meta.train
    if not prof:
        _die(f"pack {pack_name!r} has no [train] profile")
    return prof


def _invoke_prep(args, py, sd_dir, prep_py, folder, trigger, outfit, keep, prune, label):
    txts = [f for f in folder.glob("*.txt")]
    if args.skip_caption:
        plat.info(f"-skip-caption: using existing captions{label}")
    elif txts:
        plat.ok(f"captions present{label} — skipping tagger (delete .txt to re-tag)")
    else:
        plat.ok(f"auto-captioning{label} (WD14 tagger, trigger '{trigger}')")
        env = {**os.environ, "PYTHONUTF8": "1"}
        rc = plat.run([py, "finetune/tag_images_by_wd14_tagger.py", "--onnx",
                       "--repo_id", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                       "--batch_size", "4", str(folder)], cwd=sd_dir, env=env)
        if rc != 0:
            _die(f"WD14 tagger failed{label} (rc={rc})")
        prep = [py, str(prep_py), str(folder), "--trigger", trigger]
        if outfit:
            prep += ["--outfit", outfit]
        if prune:
            prep += ["--prune", prune]
        if keep:
            prep += ["--keep", keep]
        if (outfit or prune) and not args.force:
            prep.append("--strict")
        rc = plat.run(prep, env=env)
        if rc == 2:
            _die(f"outfit/prune matched NO caption tags{label} — it did NOT bake into '{trigger}'. "
                 "Fix the words in characters.toml to match the WD14 tags (open a .txt in the folder), "
                 "or re-run with --force to train anyway.")
        if rc != 0:
            _die(f"prep_captions failed{label} (rc={rc})")


def train_one(args) -> int:
    char = args.char or args.char_opt
    if not char:
        _die("no character given (usage: dev train <char>)")

    lt = _lora_train_dir()
    prof = _pack_train_profile(args.pack)
    trainer_script = prof.get("trainer_script", "sdxl_train_network.py")
    trigger_suffix = prof.get("trigger_suffix", "char")
    sd_dir = config.ROOT / prof.get("sd_scripts", "tools/sd-scripts")
    roster_file = config.ROOT / prof.get("roster_source", "tools/lora_train/roster.json")
    base_default = config.ROOT / prof.get("base_checkpoint",
                                          "models/checkpoints/oneObsession_v19Atypical.safetensors")

    # --- roster defaults (unless overridden on the CLI) ---
    entry = {}
    if roster_file.exists():
        for e in json.loads(roster_file.read_text(encoding="utf-8")):
            if e.get("name") == char:
                entry = e
                break

    outfits = entry.get("outfits") or None
    modular = bool(outfits)
    outfit_keep = entry.get("outfit_keep", {}) if modular else {}

    trigger = args.trigger or entry.get("trigger") or f"{char}{trigger_suffix}"
    prune = args.prune if args.prune is not None else entry.get("prune", "")
    if modular:
        outfit, keep = "", ""
    else:
        outfit = args.outfit if args.outfit is not None else entry.get("outfit", "")
        keep = args.keep if args.keep is not None else entry.get("keep", "")

    base = Path(args.base) if args.base else base_default
    py = venv.python("trainer")
    acc = venv.bin("trainer", "accelerate")
    if not args.dry_run and not py.exists():
        _die("trainer venv missing. Run: ./dev setup --with-trainer")

    cfg = _resolve_cfg(char, args)
    min_images = int(cfg["min_images"])

    # --- preflight (soft under --dry-run) ---
    data = config.ROOT / "output" / "dataset" / char
    out_dir = config.MODELS / "loras"
    subsets = []
    if not args.dry_run and not base.exists():
        _die(f"checkpoint not found: {base}  (pass --base <path>)")
    if args.dry_run and not base.exists():
        plat.warn(f"checkpoint not found: {base} (pass --base <path>)")

    if modular:
        for oname, garments in outfits.items():
            odir = data / oname
            ocnt = _count_images(odir)
            okeep = outfit_keep.get(oname, "") if isinstance(outfit_keep, dict) else ""
            subsets.append({"name": oname, "garments": str(garments), "keep": okeep,
                            "dir": odir, "count": ocnt})
        plat.ok(f"modular character '{char}': {len(subsets)} outfits "
                f"({', '.join(s['name'] for s in subsets)})")
        for s in subsets:
            if s["count"] < min_images:
                if args.dry_run:
                    plat.warn(f"outfit '{s['name']}': {s['count']} images in {s['dir']} "
                              f"(need >= {min_images})")
                    if s["count"] == 0:
                        s["count"] = min_images  # placeholder so the preview can plan
                else:
                    _die(f"outfit '{s['name']}': only {s['count']} images in {s['dir']} "
                         f"(need >= {min_images}). Generate with IL_DatasetEdit_{char}_{s['name']}.")
            else:
                plat.ok(f"  {s['name']}: {s['count']} images")
        img_count = sum(s["count"] for s in subsets)
    else:
        img_count = _count_images(data)
        if not data.is_dir():
            if args.dry_run:
                plat.warn(f"no dataset at {data} — using min_images ({min_images}) as a placeholder")
                img_count = min_images
            else:
                _die(f"no dataset at {data} — generate with IL_DatasetEdit (prefix 'dataset/{char}')")
        elif img_count < min_images:
            if args.dry_run:
                plat.warn(f"only {img_count} images in {data} (need >= {min_images})")
            else:
                _die(f"only {img_count} images in {data} (need >= {min_images}). Generate/curate more.")
        else:
            plat.ok(f"{img_count} images in output/dataset/{char}")

    # --- caption (skipped on --dry-run) ---
    prep_py = lt / "prep_captions.py"
    if args.dry_run:
        plat.info("-dry-run: skipping captioning")
    elif modular:
        for s in subsets:
            _invoke_prep(args, py, sd_dir, prep_py, s["dir"],
                         f"{trigger}, {char}_{s['name']}", s["garments"], s["keep"], prune,
                         f" ['{s['name']}']")
    else:
        _invoke_prep(args, py, sd_dir, prep_py, data, trigger, outfit, keep, prune, "")

    # --- write the dataset config ---
    cache_dir = lt / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cache_dir / f"{char}.toml"
    if modular:
        import dataset_plan
        counts = [s["count"] for s in subsets]
        repeats = dataset_plan.balanced_repeats(counts, cfg["steps"], cfg["epochs"], cfg["batch"])
        subs = [{"image_dir": str(s["dir"]).replace("\\", "/"), "num_repeats": r}
                for s, r in zip(subsets, repeats)]
        plan_cfg = {"resolution": cfg["resolution"], "batch": cfg["batch"],
                    "min_bucket_reso": cfg["min_bucket_reso"], "max_bucket_reso": cfg["max_bucket_reso"],
                    "bucket_reso_steps": cfg["bucket_reso_steps"]}
        toml_text = dataset_plan.render_dataset_toml(char, subs, plan_cfg, keep_tokens=2)
        plat.ok(f"{img_count} imgs across {len(subsets)} outfits -> multi-subset config "
                f"(keep_tokens=2, balanced num_repeats, target {cfg['steps']} steps)")
    else:
        repeats, actual, dev = cmd.derive_repeats(img_count, cfg["steps"], cfg["epochs"], cfg["batch"])
        plat.ok(f"{img_count} imgs x {repeats} repeats x {cfg['epochs']} epochs / batch {cfg['batch']} "
                f"~= {actual} steps (target {cfg['steps']})")
        if dev > 0.25:
            plat.warn(f"derived ~{actual} steps is {dev:.0%} off target {cfg['steps']}; "
                      "adjust --steps/--epochs if it matters")
        toml_text = cmd.render_single_subset_toml(cfg, data, repeats)
    cfg_path.write_text(toml_text, encoding="utf-8")  # UTF-8 no BOM (Python default)

    # --- accelerate command ---
    output_name = f"{char}_v1"
    train_args = cmd.build_accelerate_args(cfg, base=base, out_dir=out_dir, output_name=output_name,
                                           dataset_cfg=cfg_path, trainer_script=trainer_script)
    cfg_json = json.dumps({k: v for k, v in cfg.items() if not k.startswith("__")}, indent=2)

    if args.dry_run:
        plat.heading(f"DRY RUN ({char})")
        print("resolved params (train.toml defaults < per-char < profile < CLI):")
        print(cfg_json)
        print(f"\ndataset config ({cfg_path}):")
        print(toml_text)
        print("\naccelerate command:")
        print(f"{acc} {' '.join(train_args)}")
        plat.info("dry-run: nothing trained.")
        return 0

    lora_path = out_dir / f"{output_name}.safetensors"
    if lora_path.exists():
        plat.warn(f"models/loras/{output_name}.safetensors exists — this run will overwrite it")

    out_dir.mkdir(parents=True, exist_ok=True)
    plat.ok(f"training {output_name} (dim {cfg['dim']}/alpha {cfg['alpha']}, "
            f"{cmd.optimizer_desc(cfg)}) -> models/loras/{output_name}.safetensors")
    env = {**os.environ, "PYTHONUTF8": "1"}
    rc = plat.run([acc, *train_args], cwd=sd_dir, env=env)

    # provenance: resolved params + exact command next to the LoRA
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    (out_dir / f"{output_name}.args.txt").write_text(
        f"# dev train run for '{char}' on {stamp}\n# resolved params:\n{cfg_json}\n\n"
        f"# command:\n{acc} {' '.join(train_args)}\n", encoding="utf-8")

    if rc != 0:
        _die(f"training failed (rc={rc})")
    plat.ok(f"done. Provenance: models/loras/{output_name}.args.txt")
    plat.ok("Pick the best epoch via an XY plot of strength {0.5,0.75,0.9} x seeds in IL_1_Base.")
    return 0


def run_all(args) -> int:
    prof = _pack_train_profile(args.pack)
    roster_file = config.ROOT / prof.get("roster_source", "tools/lora_train/roster.json")
    if not roster_file.exists():
        plat.err("no roster.json — run: dev build first")
        return 1
    roster = json.loads(roster_file.read_text(encoding="utf-8"))
    total = len(roster)
    trained, skipped = 0, []
    for c in roster:
        name = c["name"]
        data = config.ROOT / "output" / "dataset" / name
        n = _count_images(data)
        if n < 12:
            plat.warn(f"skip {name}: {n} images in output/dataset/{name}/ (generate + curate first)")
            skipped.append(name)
            continue
        plat.heading(f"training {name}")
        one = copy.copy(args)
        one.char, one.char_opt, one.all = name, None, False
        try:
            train_one(one)
            trained += 1
        except TrainError as e:
            plat.err(f"{name} failed: {e}")
    plat.ok(f"trained {trained} / {total} roster characters.")
    if skipped:
        plat.info(f"skipped (no dataset yet): {', '.join(skipped)}")
    return 0


def main(argv) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.all:
            return run_all(args)
        return train_one(args)
    except TrainError as e:
        plat.err(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

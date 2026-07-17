"""Argparse dispatcher for `python -m devtools`. STDLIB ONLY.

Every subcommand's implementation is imported LAZILY inside its branch, so importing this module
(and running `dev setup`) never pulls in a torch-touching module. The test suite enforces that
(tools/tests/test_stdlib_only.py).
"""
from __future__ import annotations

import argparse


def _add_color_flag(p):
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dev", description="ComfyUI dev toolkit")
    _add_color_flag(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="provision the ComfyUI env (venv, torch, custom-node deps)")
    _add_color_flag(p_setup)
    p_setup.add_argument("--gpu", default="nvidia",
                         help="nvidia | amd-rdna3 | amd-rdna35 | amd-rdna4 | intel-xpu | cpu")
    p_setup.add_argument("--python", metavar="VER",
                         help="python for the main venv, built via uv (e.g. 3.12); "
                              "default: system python, or 3.12 if system python is 3.13+")
    p_setup.add_argument("--skip-torch", action="store_true", help="don't (re)install torch")
    p_setup.add_argument("--with-trainer", action="store_true", help="also build the LoRA trainer venv")

    p_run = sub.add_parser("run", help="launch ComfyUI")
    _add_color_flag(p_run)
    p_run.add_argument("rest", nargs=argparse.REMAINDER, help="extra args forwarded to main.py")

    p_verify = sub.add_parser("verify", help="post-install smoke checks")
    _add_color_flag(p_verify)

    p_build = sub.add_parser("build", help="generate a pack's ComfyUI workflows")
    _add_color_flag(p_build)
    p_build.add_argument("pack", nargs="?", default="il_graphs")

    p_models = sub.add_parser("models", help="download a pack's model stack")
    _add_color_flag(p_models)
    msub = p_models.add_subparsers(dest="models_cmd", required=True)
    p_mi = msub.add_parser("install")
    p_mi.add_argument("pack", nargs="?", default="il_graphs")
    p_mi.add_argument("--variant", action="append", default=[], metavar="group=value",
                      help="pick a variant group choice, e.g. --variant quant=Q5_K_M (repeatable)")
    p_mi.add_argument("--with-optional", action="store_true", help="also fetch entries marked optional")

    p_train = sub.add_parser("train", help="train a character LoRA (see `dev train --help`)")
    _add_color_flag(p_train)
    p_train.add_argument("rest", nargs=argparse.REMAINDER)

    p_val = sub.add_parser("validate", help="validate a workflow JSON against its rules")
    _add_color_flag(p_val)
    p_val.add_argument("workflow")

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    from .core import platform as plat
    if getattr(args, "no_color", False):
        plat.set_color(False)

    if args.cmd == "setup":
        from . import setup
        return setup.run(args)
    if args.cmd == "run":
        from . import run as run_cmd
        return run_cmd.run(args)
    if args.cmd == "verify":
        from .setup import verify
        return verify.run(args)
    if args.cmd == "build":
        from . import packs
        return packs.build(args.pack)
    if args.cmd == "models":
        from . import packs
        variants = {}
        for item in args.variant:
            if "=" not in item:
                ap.error(f"--variant expects group=value, got {item!r}")
            k, v = item.split("=", 1)
            variants[k.strip()] = v.strip()
        return packs.models_install(args.pack, variants=variants, with_optional=args.with_optional)
    if args.cmd == "train":
        from . import train
        return train.main(args.rest)
    if args.cmd == "validate":
        from . import packs
        return packs.validate(args.workflow)
    return 1

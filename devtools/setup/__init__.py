"""`dev setup` — one-shot, idempotent provisioning for a fresh checkout. STDLIB ONLY.

Ports setup.bat: prereqs -> submodules -> venv -> core reqs -> torch -> custom-node reqs ->
[trainer] -> verify. Runs on the bare SYSTEM python (it is what creates the venv); every pip
install targets the venv interpreter explicitly, so no shell activation is needed.
"""
from __future__ import annotations

import sys

from ..core import config
from ..core import platform as plat
from ..core import venv


def _prereqs(gpu_mode) -> bool:
    plat.heading("[1/6] Verifying prerequisites")
    plat.ok(f"python {sys.version.split()[0]} ({sys.executable})")

    if not plat.have("git"):
        plat.err("git not found on PATH")
        return False
    rc, out = plat.capture(["git", "--version"])
    plat.ok(out.strip() or "git present")

    if plat.have("ssh"):
        plat.step("testing ssh -T git@github.com (may prompt to accept the host key on first run)")
        rc, _ = plat.capture(["ssh", "-T", "-o", "BatchMode=yes",
                              "-o", "StrictHostKeyChecking=accept-new", "git@github.com"])
        rc2, out2 = plat.capture(["ssh", "-T", "git@github.com"])
        if "successfully authenticated" in out2:
            plat.ok("SSH to github.com works")
        else:
            plat.warn("SSH auth to github.com not confirmed — HTTPS remotes still work; "
                      "set up a key if submodules use git@ URLs")
    else:
        plat.warn("ssh not found — skipping github.com auth check")

    if plat.have("nvidia-smi"):
        rc, out = plat.capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        for line in out.splitlines():
            if line.strip():
                plat.ok(f"GPU: {line.strip()}")
    else:
        plat.warn("nvidia-smi not found — GPU optional but expected for production gen")
    return True


def _submodules() -> bool:
    plat.heading("[2/6] Initializing submodules (first run can take a few minutes)")
    rc = plat.run(["git", "submodule", "update", "--init", "--recursive", "--jobs", "4"],
                  cwd=config.ROOT)
    if rc != 0:
        plat.warn("some submodules failed; retrying with --depth 1")
        rc = plat.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
                       "--jobs", "4"], cwd=config.ROOT)
        if rc != 0:
            plat.err("submodule init failed")
            return False
    plat.ok("submodules initialized")
    return True


def _venv(python_spec) -> bool:
    plat.heading("[3/6] Setting up venv")
    if not venv.ensure_main(python_spec=python_spec):
        return False
    py = venv.python("main")
    plat.run([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    plat.ok("pip/wheel/setuptools upgraded")
    return True


def _core_reqs_and_torch(gpu_mode, skip_torch) -> bool:
    plat.heading("[4/6] Core requirements + torch")
    py = venv.python("main")
    req = config.ROOT / "requirements.txt"
    if req.exists():
        # only-if-needed: upgrade pinned packages (frontend etc.) without clobbering CUDA torch
        # with the unpinned PyPI CPU wheel that requirements.txt would otherwise pull.
        rc = plat.run([py, "-m", "pip", "install", "--upgrade",
                       "--upgrade-strategy", "only-if-needed", "-r", str(req)])
        if rc != 0:
            plat.err("ComfyUI requirements install failed")
            return False
        plat.ok("ComfyUI requirements installed/upgraded")
    else:
        plat.warn("requirements.txt missing — skipping core install")

    if skip_torch:
        plat.info("skipping torch install per --skip-torch")
        return True
    from . import torch as torch_mod
    if torch_mod.install(gpu_mode, py) != 0:
        plat.warn("torch install reported issues — check output above")
    return True


def _node_reqs() -> bool:
    plat.heading("[5/6] Custom-node requirements")
    from . import node_reqs
    if node_reqs.install(venv.python("main")) != 0:
        plat.warn("some custom-node installs failed — see log above")
    return True


def _trainer() -> None:
    plat.heading("[+] Provisioning LoRA trainer venv (sd-scripts, Blackwell torch)")
    from . import trainer
    if trainer.install() != 0:
        plat.warn("trainer venv install reported issues — see above")


def run(args) -> int:
    if getattr(args, "no_color", False):
        plat.set_color(False)
    print(plat.c("\n================================================", "cyan"))
    print(plat.c(" ComfyUI Setup", "cyan"))
    print(plat.c(f" Root: {config.ROOT}", "cyan"))
    print(plat.c("================================================", "cyan"))

    # Pick the main-venv interpreter: an explicit --python wins; otherwise, if the system python is
    # too new for the ML stack (3.13+) and uv is available, pin 3.12 so torch/custom-node wheels exist.
    python_spec = args.python
    if python_spec is None and plat.have("uv") and sys.version_info[:2] >= (3, 13):
        python_spec = "3.12"
        plat.info(f"system python is {sys.version.split()[0]} (very new for the ML stack) — building "
                  "the main venv with uv python 3.12 (pass --python to override, e.g. --python 3.13)")

    if not _prereqs(args.gpu):
        return _fail()
    if not _submodules():
        return _fail()
    if not _venv(python_spec):
        return _fail()
    if not _core_reqs_and_torch(args.gpu, args.skip_torch):
        return _fail()
    _node_reqs()
    if args.with_trainer:
        _trainer()

    plat.heading("[6/6] Verifying install")
    from . import verify
    verify_rc = verify.run(args)

    if verify_rc != 0:
        plat.heading("Setup completed with WARNINGS — see above")
    else:
        plat.heading("Setup complete!")
        print("\nNext steps:")
        print("  1. Workflows + models live in a separate repo — see ONBOARDING.md")
        print("  2. ./dev run                     — launch ComfyUI")
        if args.with_trainer:
            print("  3. tools/lora_train/README.md    — generate data, caption, train a LoRA")
    return 0


def _fail() -> int:
    plat.heading("Setup FAILED — fix errors above and re-run")
    return 1

"""Create the LoRA-training venv (kohya sd-scripts) with Blackwell-ready deps. STDLIB ONLY.

Port of scripts/install_trainer.ps1. Independent of the main venv: sd-scripts needs Python <=3.12
and a pinned torch, so this builds a SEPARATE venv at tools/lora_train/.venv via `uv` (which pins
py3.11). The trainer CODE is the tools/sd-scripts submodule; this only provisions its Python env.
Idempotent; safe to re-run.
"""
from __future__ import annotations

from ..core import config
from ..core import platform as plat
from ..core import venv

TORCH_VERSION = "2.7.0"   # sd-scripts-compatible, Blackwell-capable
CUDA_TAG = "cu128"        # RTX 5080 (sm_120) needs CUDA 12.8+


def install(torch_version=TORCH_VERSION, cuda_tag=CUDA_TAG) -> int:
    sd_dir = config.TOOLS / "sd-scripts"
    venv_dir = venv.path("trainer")
    py = venv.python("trainer")
    reqs = sd_dir / "requirements.txt"

    if not reqs.exists():
        plat.err("tools/sd-scripts not initialized. Run: git submodule update --init tools/sd-scripts")
        return 1
    if not plat.have("uv"):
        plat.err("uv not found. Install it (https://docs.astral.sh/uv/), then re-run. uv pins Python 3.11.")
        return 1

    # 1. venv (idempotent)
    if not py.exists():
        plat.step("creating trainer venv (Python 3.11) at tools/lora_train/.venv")
        if plat.run(["uv", "venv", "--python", "3.11", str(venv_dir)]) != 0:
            plat.err("uv venv failed")
            return 1
    else:
        plat.ok("trainer venv already exists")

    # 2. torch (skip if already a CUDA build — mirrors the main torch install)
    rc, out = plat.capture([py, "-c", "import torch; print(torch.version.cuda or 'cpu')"])
    cuda = out.strip().splitlines()[-1] if out.strip() else ""
    if rc == 0 and cuda and cuda != "cpu":
        plat.ok(f"trainer torch already CUDA-enabled (cuda={cuda}) — skipping reinstall")
    else:
        plat.step(f"installing torch {torch_version} + torchvision ({cuda_tag}) for Blackwell")
        if plat.run(["uv", "pip", "install", "--python", str(py), f"torch=={torch_version}",
                     "torchvision", "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"]) != 0:
            plat.err("torch install failed")
            return 1

    # 3. sd-scripts requirements (run from sd-scripts dir so the `-e .` line resolves there)
    plat.step(f"installing sd-scripts requirements (from {sd_dir})")
    if plat.run(["uv", "pip", "install", "--python", str(py), "-r", "requirements.txt"],
                cwd=sd_dir) != 0:
        plat.warn("some sd-scripts requirements failed — see above")

    # onnx/onnxruntime power the WD14 tagger (CPU is plenty and avoids Blackwell EP issues);
    # prodigyopt is the default optimizer.
    plat.step("installing tagger + optimizer deps (onnxruntime, onnx, prodigyopt)")
    if plat.run(["uv", "pip", "install", "--python", str(py),
                 "onnxruntime", "onnx", "prodigyopt"]) != 0:
        plat.warn("tagger/optimizer deps reported issues")

    # 4. accelerate default config (non-interactive) + GPU visibility check
    acc = venv.bin("trainer", "accelerate")
    if acc.exists():
        plat.run([acc, "config", "default"])
    rc, out = plat.capture(
        [py, "-c", "import torch; print(torch.cuda.get_device_name(0) "
                   "if torch.cuda.is_available() else 'NO CUDA')"])
    plat.ok(f"trainer venv torch sees: {out.strip().splitlines()[-1] if out.strip() else '?'}")
    plat.ok("trainer ready. Next: tools/lora_train/README.md (generate data -> caption -> train).")
    return 0

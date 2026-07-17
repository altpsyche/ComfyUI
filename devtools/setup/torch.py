"""Install torch + torchvision into a venv for the requested GPU stack. STDLIB ONLY.

Port of scripts/install_torch.ps1. Maps a GPU mode to the right PyTorch index URL (+ --pre for the
ROCm nightlies), then force-reinstalls torch+torchvision to overwrite any CPU wheel that
`pip install -r requirements.txt` may have left behind.
"""
from __future__ import annotations

from ..core import platform as plat

# GPU mode -> (index url, needs --pre). None index for nvidia (resolved dynamically).
_MODES = {
    "amd-rdna3":  ("https://rocm.nightlies.amd.com/v2/gfx110X-all/", True),
    "amd-rdna35": ("https://rocm.nightlies.amd.com/v2/gfx1151/", True),
    "amd-rdna4":  ("https://rocm.nightlies.amd.com/v2/gfx120X-all/", True),
    "intel-xpu":  ("https://download.pytorch.org/whl/xpu", False),
    "cpu":        ("https://download.pytorch.org/whl/cpu", False),
}


def resolve_nvidia_cuda_major():
    """Pick a cuXXX tag from the installed NVIDIA driver version (matches install_torch.ps1)."""
    rc, out = plat.capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if rc != 0 or not out.strip():
        return None
    try:
        major = int(out.strip().splitlines()[0].split(".")[0])
    except (ValueError, IndexError):
        return None
    if major >= 580:
        return "130"
    if major >= 555:
        return "128"
    if major >= 525:
        return "121"
    if major >= 470:
        return "118"
    return None


def install(gpu_mode, py) -> int:
    index_url = None
    pre = False
    label = gpu_mode

    if gpu_mode == "nvidia":
        cuda = resolve_nvidia_cuda_major()
        if cuda:
            index_url = f"https://download.pytorch.org/whl/cu{cuda}"
            label = f"nvidia cu{cuda}"
        else:
            plat.warn("nvidia-smi unavailable; falling back to the CPU wheel")
            index_url = "https://download.pytorch.org/whl/cpu"
            label = "cpu (nvidia-smi missing)"
    elif gpu_mode in _MODES:
        index_url, pre = _MODES[gpu_mode]
    else:
        plat.err(f"unknown --gpu mode: {gpu_mode}. "
                 "Valid: nvidia, amd-rdna3, amd-rdna35, amd-rdna4, intel-xpu, cpu")
        return 1

    plat.step(f"torch install: {label}  (index: {index_url})")

    # Fast path: for nvidia, if torch is already a CUDA build, don't reinstall.
    if gpu_mode == "nvidia":
        rc, out = plat.capture([py, "-c", "import torch; print(torch.version.cuda or 'cpu')"])
        existing = out.strip().splitlines()[-1] if out.strip() else ""
        if rc == 0 and existing and existing != "cpu":
            plat.ok(f"torch already CUDA-enabled (cuda={existing}) — skipping reinstall")
            return 0

    cmd = [py, "-m", "pip", "install", "--force-reinstall"]
    if pre:
        cmd.append("--pre")
    cmd += ["torch", "torchvision", "--index-url", index_url]
    if plat.run(cmd) != 0:
        plat.warn(f"torch install failed — install manually with: {' '.join(str(x) for x in cmd)}")
        return 1
    plat.ok(f"torch + torchvision installed for {label}")
    return 0

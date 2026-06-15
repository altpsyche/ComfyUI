"""Verify the LoRA-training venv actually works (not just that torch imports).
Run with the trainer venv python:  tools/lora_train/.venv/Scripts/python.exe verify_env.py
"""
import importlib
import math
import pathlib
import sys

ok = True


def check(name, fn):
    global ok
    try:
        fn()
        print(f"[+] {name}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[x] {name}: {type(e).__name__}: {e}")


import torch  # noqa: E402

avail = torch.cuda.is_available()
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"device: {torch.cuda.get_device_name(0) if avail else 'NO CUDA'} | "
      f"cc {torch.cuda.get_device_capability(0) if avail else '-'}")


def gpu_compute():
    assert torch.cuda.is_available(), "cuda not available to torch"
    # bf16 matmul = the training dtype; proves sm_120 kernels exist (Blackwell gotcha)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    assert math.isfinite(y), "non-finite GPU result"


check("GPU bf16 matmul (sm_120 kernels present)", gpu_compute)
check("sd-scripts library.train_util import", lambda: importlib.import_module("library.train_util"))
check("sd-scripts sdxl_train_util import", lambda: importlib.import_module("library.sdxl_train_util"))
check("prodigyopt optimizer", lambda: importlib.import_module("prodigyopt"))
check("accelerate", lambda: importlib.import_module("accelerate"))


def onnx_ok():
    import onnxruntime as ort
    assert ort.get_available_providers(), "no onnxruntime providers"


check("onnxruntime (WD14 tagger)", onnx_ok)


def dataset_cfg():
    import toml
    d = toml.load(pathlib.Path(__file__).parent / "dataset_charA.toml")
    assert d["datasets"][0]["resolution"] == 1024


check("dataset_charA.toml parses", dataset_cfg)

print("\nRESULT:", "ALL GOOD" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

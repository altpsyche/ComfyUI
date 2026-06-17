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
    # Parse a dataset config shaped exactly like the one train_lora.ps1 generates into .cache/<char>.toml.
    # Deliberately uses the third-party `toml` package (NOT stdlib tomllib): that is the reader sd-scripts
    # itself uses, so this proves a real training dependency is installed and parses the config we feed it.
    import toml
    sample = (
        '[general]\nshuffle_caption = true\nkeep_tokens = 1\ncaption_extension = ".txt"\n'
        '[[datasets]]\nresolution = 1024\nbatch_size = 2\nenable_bucket = true\n'
        'min_bucket_reso = 768\nmax_bucket_reso = 1280\nbucket_reso_steps = 64\n'
        '  [[datasets.subsets]]\n  image_dir = "x"\n  num_repeats = 10\n'
    )
    d = toml.loads(sample)
    assert d["datasets"][0]["resolution"] == 1024


check("dataset config TOML parses (sd-scripts 'toml' reader)", dataset_cfg)


def dataset_wildcards():
    # The IL_DatasetEdit generator drives variety through these six wildcard files; without them the
    # edit instruction emits literal __pose__ tokens. Repo root = three levels up from this file.
    wdir = pathlib.Path(__file__).resolve().parents[2] / "custom_nodes/ComfyUI-Impact-Pack/wildcards"
    need = ["angle", "pose", "expression", "framing", "background", "lighting"]
    missing = [w for w in need if not (wdir / f"{w}.txt").exists()]
    assert not missing, f"missing dataset wildcards in {wdir}: {missing}"


check("dataset wildcards present (angle/pose/expression/framing/background/lighting)", dataset_wildcards)

print("\nRESULT:", "ALL GOOD" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

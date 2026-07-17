"""Post-install smoke checks. Port of scripts/verify.ps1.

Two entry points:
  run(args)     — stdlib launcher (system python) that re-execs into the MAIN venv, because the
                  checks import torch + comfy_script which only exist there.
  _run_checks() — the actual checks, executed as `python -m devtools.setup.verify --run`.
"""
from __future__ import annotations

import sys

from ..core import config
from ..core import platform as plat
from ..core import venv


def run(args=None) -> int:
    """Launcher: hand off to the main venv's python to run the real checks."""
    if not venv.exists("main"):
        plat.err("main venv missing — run: dev setup")
        return 1
    return venv.reexec("main", ["-m", "devtools.setup.verify", "--run"], cwd=config.ROOT)


def _check(name, fn) -> bool:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"  ... {name} {plat.c(f'FAIL ({type(e).__name__}: {e})', 'red')}")
        return False
    print(f"  ... {name} {plat.c('OK', 'green')}")
    return True


def _run_checks() -> int:
    passed = failed = 0

    def tally(ok_):
        nonlocal passed, failed
        if ok_:
            passed += 1
        else:
            failed += 1

    def torch_gpu():
        import torch
        print(f" (cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()})", end="")

    tally(_check("torch GPU check", torch_gpu))

    def comfyscript_load():
        from comfy_script.runtime import load
        load()

    tally(_check("ComfyScript virtual-mode load", comfyscript_load))

    def key_packs():
        need = ["ComfyUI-Manager", "ComfyUI-Impact-Pack", "rgthree-comfy", "was-ns", "ComfyScript"]
        miss = [d for d in need if not (config.CUSTOM_NODES / d).is_dir()]
        if miss:
            raise AssertionError(f"missing packs: {miss}")

    tally(_check("key custom packs present on disk", key_packs))

    from ..core import nodes

    def submodules_clean():
        bad = [f"{p} [{st}]" for p, (_, st) in nodes.submodule_shas().items() if st in "+-U"]
        if bad:
            raise AssertionError("dirty/uninitialized: " + ", ".join(bad))

    tally(_check("git submodule status clean", submodules_clean))

    print(f"\n  Pass: {passed}  Fail: {failed}")
    print("\n  Pinned submodule SHAs:")
    for path_, (sha, st) in sorted(nodes.submodule_shas().items()):
        print(f"    {st}{sha} {path_}")
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--run" in args:
        sys.exit(_run_checks())
    sys.exit(run())

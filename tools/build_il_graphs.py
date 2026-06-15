"""Generate the tiered IL_*/SDXL ComfyUI workflow family.

The generator was split into the `il_graphs` package (config / templates / builder /
layers / graphs / docs / build). This file stays as the entrypoint.

Run:  python tools/build_il_graphs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from il_graphs.build import main  # noqa: E402

if __name__ == "__main__":
    main()

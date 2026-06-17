"""Shared pytest setup for the LoRA-tooling tests.

Puts tools/ and tools/lora_train/ on sys.path so the generator package and the trainer helpers import
by name, and provides a session-scoped `built` fixture that regenerates the IL_* workflows once.
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]          # tools/
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "lora_train"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def built():
    """Run the workflow generator once; return the output dir (config.OUT)."""
    import il_graphs.build as B
    saved = sys.argv
    sys.argv = ["build", "--no-validate"]            # validate separately in test_build
    try:
        B.main()
    finally:
        sys.argv = saved
    from il_graphs.config import OUT
    return OUT

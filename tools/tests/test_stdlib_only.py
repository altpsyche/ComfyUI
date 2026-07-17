"""The dispatcher + core + setup path must import with ZERO third-party deps, because `dev setup`
runs on a bare system python before any venv exists. `-S` disables site-packages, so any stray
third-party import at module load time fails this."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_MODULES = [
    "devtools.cli",
    "devtools.core.platform",
    "devtools.core.config",
    "devtools.core.venv",
    "devtools.core.nodes",
    "devtools.core.download",
    "devtools.setup",
    "devtools.run",
]


def test_stdlib_only_import():
    code = "import " + ", ".join(_MODULES)
    r = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"non-stdlib import leaked into the setup path:\n{r.stderr}"

"""venv exe resolution is the one place that knows Scripts\\ vs bin/ — pin both OS branches."""
from pathlib import Path

from devtools.core import platform as plat


def test_venv_bin_posix(monkeypatch):
    monkeypatch.setattr(plat, "IS_WINDOWS", False)
    assert plat.venv_bin("/x/venv", "python") == Path("/x/venv/bin/python")
    assert plat.venv_bin("/x/venv", "accelerate") == Path("/x/venv/bin/accelerate")


def test_venv_bin_windows(monkeypatch):
    monkeypatch.setattr(plat, "IS_WINDOWS", True)
    assert plat.venv_bin("C:/x/venv", "python") == Path("C:/x/venv/Scripts/python.exe")
    assert plat.venv_bin("C:/x/venv", "accelerate") == Path("C:/x/venv/Scripts/accelerate.exe")
    # already-.exe names aren't doubled
    assert plat.venv_bin("C:/x/venv", "python.exe") == Path("C:/x/venv/Scripts/python.exe")

"""What the wheel promises to install must cover what its modules import.

The GPU smoke test caught this the expensive way: `assemble.py` imports `soundfile` at module scope and
`folivox.audio` needs it too, but the wheel declared only `numpy` and Folivox keeps `soundfile` under its
`tts`/`all` extras. On this machine it was installed anyway, so every test passed and the kernel run was the
first thing to fail — with `ModuleNotFoundError: No module named 'soundfile'`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "folivox_ming"
PYPROJECT = ROOT / "pyproject.toml"

# Modules this package may assume are importable without being declared in pyproject.toml, each for a reason
# that makes the assumption safe rather than hopeful.
ALLOWED = {
    "folivox_ming": "this package itself",
    "folivox": "the host: installed separately, deliberately not a hard dependency",
    "platformdirs": "a hard requirement of folivox (not an extra), so it is present wherever the host is",
    "torch": "worker.py only, which runs in the Engine environment that Ming's env installs",
    "ming_tts": "worker.py only, from the cloned Ming repo put on the worker's sys.path",
    "pynini": "optional by design: the worker probes for it and disables text normalization when it is absent",
}
STDLIB = set(__import__("sys").stdlib_module_names)


def _imported_names(path: Path) -> set[str]:
    """Every top-level module named by an import anywhere in one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _declared() -> set[str]:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    deps = set()
    for spec in project.get("dependencies", []):
        name = spec.split(";")[0].split(">=")[0].split("==")[0].split("<")[0].split("[")[0]
        deps.add(name.strip().replace("_", "-").lower())
    return deps


def _distribution_names() -> dict[str, str]:
    """Import name -> distribution name, for the cases where they differ."""
    return {"soundfile": "soundfile", "numpy": "numpy", "huggingface_hub": "huggingface-hub"}


@pytest.mark.parametrize("source", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_every_third_party_import_is_a_declared_dependency(source: Path):
    third_party = {
        name for name in _imported_names(source)
        if name not in STDLIB and name not in ALLOWED and not name.startswith(".")
    }
    declared = _declared()
    aliases = _distribution_names()

    missing = {
        name for name in third_party
        if aliases.get(name, name).replace("_", "-").lower() not in declared
    }

    assert not missing, (
        f"{source.name} imports {sorted(missing)}, which pyproject.toml does not declare as a dependency. "
        f"A wheel built from it would fail on a host that has not installed them by hand. If the import only "
        f"exists in the Engine environment, add it to ALLOWED with the reason."
    )


def test_soundfile_is_declared():
    """The one that actually bit: Folivox's own metadata does not pull it in."""
    assert "soundfile" in _declared()

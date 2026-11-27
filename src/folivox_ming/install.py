"""Where the Ming engine and its environment live, and how they are installed.

The layout copies Folivox's own ``engines/`` tree, so both tools keep their large files in the same place:

    <user data>/folivox/engines/
        ming-3.0-tts-0.5b/
            repo/         the Ming inference code, pinned by commit (ming_tts, infer.py, requirements.txt)
            weights/      the checkpoint, downloaded from Hugging Face
            ready.json    written last: what was installed, when
        runtimes/
            ming-0.5b/venv/   the private environment (torch + transformers 4.46.3, not Folivox's)

Ming's model code is a repository, not a wheel, so it is cloned and put on the Worker's ``sys.path``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

ENGINE_ID = "ming-3.0-tts-0.5b"
RUNTIME_ID = "ming-0.5b"
REPO = "https://github.com/inclusionAI/Ming-3.0-tts"
# The checkpoint and the code that reads it are released together; pin a commit here for a reproducible install.
REF = "main"
MODEL = "inclusionAI/Ming-3.0-tts-0.5b"
PYTHON = "3.11"
READY = "ready.json"
TITLE = "Ming-3.0-tts-0.5b"
# Not in Ming's requirements, but the worker reads the model's WAV with one of them.
EXTRA = ("soundfile",)
STEP_TIMEOUT = 7200.0  # one install step: clone, pip install of torch, checkpoint download

Log = Callable[[str], None]


class InstallError(RuntimeError):
    pass


def _root() -> Path:
    override = os.environ.get("FOLIVOX_MING_HOME")
    if override:
        return Path(override).expanduser()
    from platformdirs import user_data_dir  # a Folivox dependency, so it is installed

    return Path(user_data_dir("folivox", appauthor=False)) / "engines"


def build_dir() -> Path:
    return _root() / ENGINE_ID


def repo_dir() -> Path:
    return build_dir() / "repo"


def weights_dir() -> Path:
    return build_dir() / "weights"


def venv_dir() -> Path:
    return _root() / "runtimes" / RUNTIME_ID / "venv"


def venv_python() -> Path:
    return venv_dir() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# What transformers actually loads. An interrupted download leaves config.json (it is tiny and first) without
# these, so the config alone must never count as a checkpoint — that is how a half-downloaded install reads
# as "ready" and then fails at load with "no file named pytorch_model.bin".
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                "pytorch_model.bin.index.json")


def weights_present() -> bool:
    return (weights_dir() / "config.json").is_file() and any(
        (weights_dir() / name).is_file() for name in WEIGHT_FILES)


def is_ready() -> bool:
    """Environment, code and checkpoint all in place (the marker is written only after all three)."""
    try:
        info = json.loads((build_dir() / READY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if info.get("ref") != REF or not venv_python().is_file():
        return False
    return (repo_dir() / "ming_tts.py").is_file() and weights_present()


def status() -> str:
    if is_ready():
        return f"{TITLE} is installed ({weights_dir()})"
    missing = [name for name, ok in (("environment", venv_python().is_file()),
                                     ("inference code", (repo_dir() / "ming_tts.py").is_file()),
                                     ("checkpoint", weights_present())) if not ok]
    return f"{TITLE} is not installed (missing: {', '.join(missing) or 'the ready marker'})"


def _run(cmd: list[str], what: str, log: Log) -> None:
    log(what)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=STEP_TIMEOUT)  # noqa: S603 — argv, no shell
    except subprocess.TimeoutExpired:
        raise InstallError(f"{what} did not finish in {STEP_TIMEOUT / 3600:.0f} h and was stopped") from None
    except OSError as e:
        raise InstallError(f"{what} could not start: {e}") from e
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise InstallError(f"{what} failed:\n{tail}")


def ensure_runtime(log: Log = print) -> Path:
    """Create the private environment and install Ming's requirements (torch: the long step).

    ``uv`` is used when present — it finds Python 3.11 even where the system Python is newer; otherwise the
    running interpreter must already be a version Ming supports.
    """
    py = venv_python()
    if py.is_file():
        return py
    venv = venv_dir()
    venv.parent.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if uv:
        _run([uv, "venv", "--python", PYTHON, str(venv)], f"Creating a Python {PYTHON} environment", log)
        _run([uv, "pip", "install", "--python", str(py), "-r", str(repo_dir() / "requirements.txt")],
             "Installing the Ming packages", log)
        _run([uv, "pip", "install", "--python", str(py), *EXTRA], "Installing the worker additions", log)
    else:
        if sys.version_info[:2] not in ((3, 10), (3, 11)):
            raise InstallError("`uv` is needed to create the Ming environment (https://docs.astral.sh/uv/): "
                               "this interpreter is not Python 3.10 or 3.11, which the model supports")
        _run([sys.executable, "-m", "venv", str(venv)], "Creating the Ming environment", log)
        _run([str(py), "-m", "pip", "install", "--quiet", "-r", str(repo_dir() / "requirements.txt")],
             "Installing the Ming packages", log)
        _run([str(py), "-m", "pip", "install", "--quiet", *EXTRA], "Installing the worker additions", log)
    return py


def install(log: Log = print) -> None:
    """Clone the code, build the environment, fetch the checkpoint. Safe to run again (it continues)."""
    if not shutil.which("git"):
        raise InstallError("git is needed to fetch the Ming inference code")
    if not (repo_dir() / "ming_tts.py").is_file():
        repo_dir().parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", "--branch", REF, REPO, str(repo_dir())],
             f"Fetching the Ming code ({REF})", log)
    ensure_runtime(log)
    if not weights_present():
        _run([str(venv_python()), "-c",
              "import sys; from huggingface_hub import snapshot_download; "
              f"snapshot_download({MODEL!r}, local_dir=sys.argv[1], max_workers=4)", str(weights_dir())],
             f"Downloading the checkpoint ({MODEL})", log)
    (build_dir() / READY).write_text(
        json.dumps({"ref": REF, "model": MODEL, "installed": time.time()}, indent=2), encoding="utf-8")
    log(f"{TITLE} is ready.")


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:])
    if args and args[0] in ("--check", "status"):
        print(status())
        print(f"Paths under {_root()}")
        return 0 if is_ready() else 1
    try:
        install()
    except InstallError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

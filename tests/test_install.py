"""What a fresh host learns from `python -m folivox_ming.install --check`."""

from __future__ import annotations

import json

import pytest

from folivox_ming import install


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_MING_HOME", str(tmp_path / "engines"))
    return tmp_path / "engines"


def _fabricate(*, ref: str = install.REF) -> None:
    (install.repo_dir()).mkdir(parents=True, exist_ok=True)
    (install.repo_dir() / "ming_tts.py").write_text("", encoding="utf-8")
    install.weights_dir().mkdir(parents=True, exist_ok=True)
    (install.weights_dir() / "config.json").write_text("{}", encoding="utf-8")
    install.venv_python().parent.mkdir(parents=True, exist_ok=True)
    install.venv_python().write_text("", encoding="utf-8")
    (install.build_dir() / install.READY).write_text(json.dumps({"ref": ref}), encoding="utf-8")


def test_nothing_installed_names_every_missing_part(capsys):
    code = install.main(["--check"])
    out = capsys.readouterr().out

    assert code == 1
    assert "not installed" in out
    for part in ("environment", "inference code", "checkpoint"):
        assert part in out


def test_a_complete_install_reports_ready(capsys):
    _fabricate()

    code = install.main(["--check"])

    assert code == 0
    assert "is installed" in capsys.readouterr().out


def test_a_different_pinned_ref_is_not_ready(capsys):
    _fabricate(ref="some-other-commit")

    assert install.is_ready() is False
    assert install.main(["--check"]) == 1
    assert "the ready marker" in capsys.readouterr().out

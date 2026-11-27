"""CLI seam: one command renders with Ming, then assembles at the requested reading speed."""

from __future__ import annotations

import subprocess

import pytest

from folivox_ming import cli


@pytest.fixture
def book_file(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("Chapter One\n\nA short book about nothing much.\n", encoding="utf-8")
    return path


@pytest.fixture
def stub_render(monkeypatch):
    """Replace synthesis: no GPU, no checkpoint, no model — but the work folder is real."""
    calls: list = []

    def fake(book, opts, events=None):
        calls.append((book.title, opts.tts, opts.speed))
        return opts.output

    monkeypatch.setattr(cli, "render", fake)
    return calls


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def test_help_documents_the_flags_and_where_speed_applies(capsys):
    with pytest.raises(SystemExit):
        _run(["render", "--help"])
    text = capsys.readouterr().out
    for flag in ("--out", "--work-dir", "--speed", "--device", "--voice"):
        assert flag in text
    assert "assembly" in text.lower()


def test_renders_once_then_writes_a_chaptered_book(tmp_path, book_file, work_folder, stub_render):
    out = tmp_path / "book.m4b"

    code = _run(["render", str(book_file), "--out", str(out), "--work-dir", str(work_folder.dir)])

    assert code == 0
    assert len(stub_render) == 1
    assert stub_render[0][1] == "ming"
    assert stub_render[0][2] == 1.0  # speed never reaches the render, per ADR 0002
    marks = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_entries",
                            "chapter", str(out)], capture_output=True, text=True, encoding="utf-8")
    assert marks.returncode == 0
    assert marks.stdout.count('"id"') == len(work_folder.chapters)


def test_a_second_speed_reuses_the_takes(tmp_path, book_file, work_folder, stub_render):
    def mtimes():
        return {p.name: p.stat().st_mtime_ns for p in sorted(work_folder.takes_dir.glob("*.flac"))}

    before = mtimes()
    _run(["render", str(book_file), "--out", str(tmp_path / "a.m4b"), "--work-dir", str(work_folder.dir),
          "--speed", "1.15"])
    _run(["render", str(book_file), "--out", str(tmp_path / "b.m4b"), "--work-dir", str(work_folder.dir),
          "--speed", "0.9"])

    assert mtimes() == before
    assert (tmp_path / "a.chapters" / "speed-1.15").is_dir()
    assert (tmp_path / "b.chapters" / "speed-0.9").is_dir()


def test_impossible_speed_fails_before_rendering(tmp_path, book_file, work_folder, stub_render, capsys):
    code = _run(["render", str(book_file), "--out", str(tmp_path / "book.m4b"),
                 "--work-dir", str(work_folder.dir), "--speed", "0"])

    assert code != 0
    assert stub_render == []
    assert "speed" in capsys.readouterr().err.lower()

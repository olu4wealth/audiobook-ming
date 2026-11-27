"""Seam 2: Assembly's other half — the chaptered M4B, whose marks follow the stretched audio."""

from __future__ import annotations

import json
import subprocess

import pytest
from folivox.audio.assemble import BookMeta

from folivox_ming.assemble import stretch_chapters, write_audiobook

META = BookMeta(title="Fixture Book", author="A. Writer", narrator="en-kalynda")


def _probe(path, entries: str = "chapter") -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_entries", entries,
                           str(path)], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_one_chapter_mark_per_chapter_in_order_with_titles(tmp_path, work_folder):
    out = tmp_path / "book.m4b"
    chapters = stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.0)

    write_audiobook(chapters, out, META)

    marks = _probe(out)["chapters"]
    assert [m["tags"]["title"] for m in marks] == work_folder.chapters
    assert out.stat().st_size > 0


def test_marks_follow_the_stretched_durations(tmp_path, work_folder):
    out = tmp_path / "fast.m4b"
    chapters = stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.15)

    write_audiobook(chapters, out, META)

    marks = _probe(out)["chapters"]
    starts = [float(m["start_time"]) for m in marks]
    assert starts[0] == pytest.approx(0.0, abs=0.1)
    accumulated = sum(c.duration for c in chapters[:-1])
    assert starts[-1] == pytest.approx(accumulated, abs=0.15)


def test_book_metadata_reaches_the_file(tmp_path, work_folder):
    out = tmp_path / "book.m4b"
    write_audiobook(stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.0), out, META)

    tags = _probe(out, "format_tags")["format"]["tags"]
    assert tags["title"].lower() == "fixture book"
    assert tags["artist"].lower() == "a. writer"


def test_cover_is_optional_and_attached_when_given(tmp_path, work_folder):
    chapters = stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.0)
    plain = tmp_path / "plain.m4b"
    write_audiobook(chapters, plain, META)
    assert not [s for s in _probe(plain, "stream")["streams"] if s["codec_type"] == "video"]

    cover = tmp_path / "cover.jpg"
    _write_cover(cover)
    with_art = tmp_path / "art.m4b"
    write_audiobook(chapters, with_art, META, cover=cover.read_bytes())

    streams = _probe(with_art, "stream")["streams"]
    assert sum(1 for s in streams if s["codec_type"] == "video") == 1


def test_unrendered_chapter_never_becomes_a_mark(tmp_path, work_folder_factory):
    partial = work_folder_factory(["Opening", "The Long Road"], rendered=1)
    chapters = stretch_chapters(partial.dir, tmp_path / "out", speed=1.0)
    out = tmp_path / "partial.m4b"

    write_audiobook(chapters, out, META)

    assert [m["tags"]["title"] for m in _probe(out)["chapters"]] == ["Opening"]


def _write_cover(path) -> None:
    """A square cover, produced with the same ffmpeg the assembly uses."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                           "color=c=teal:s=256x256", "-frames:v", "1", str(path)],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr

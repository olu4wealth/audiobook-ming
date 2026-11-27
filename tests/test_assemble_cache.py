"""Seam 2: Assembly is cheap to repeat, and refuses input it cannot honour."""

from __future__ import annotations

import os

import pytest
from folivox.audio.assemble import BookMeta

from folivox_ming.assemble import assemble

META = BookMeta(title="Fixture Book")


class Warnings:
    def __init__(self):
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(message)


def _snapshot(folder) -> dict[str, int]:
    return {p.name: int(p.stat().st_mtime_ns) for p in sorted(folder.glob("*.flac"))}


def test_second_run_at_the_same_speed_rebuilds_nothing(tmp_path, work_folder):
    out = tmp_path / "book.m4b"

    first = assemble(work_folder.dir, out, speed=1.15, meta=META)
    chapters_dir = first.parent / (first.stem + ".chapters") / "speed-1.15"
    before = _snapshot(chapters_dir)

    assemble(work_folder.dir, out, speed=1.15, meta=META)

    assert _snapshot(chapters_dir) == before


def test_a_new_speed_keeps_the_earlier_speed_s_audio(tmp_path, work_folder):
    out = tmp_path / "book.m4b"

    assemble(work_folder.dir, out, speed=1.15, meta=META)
    assemble(work_folder.dir, out, speed=1.0, meta=META)

    root = out.parent / (out.stem + ".chapters")
    assert (root / "speed-1.15").is_dir() and (root / "speed-1").is_dir()
    assert len(list((root / "speed-1.15").glob("*.flac"))) == len(work_folder.chapters)


@pytest.mark.parametrize("speed", [0, -1, float("nan")])
def test_impossible_speeds_are_refused_before_any_audio_is_written(tmp_path, work_folder, speed):
    out = tmp_path / "book.m4b"

    with pytest.raises(ValueError, match="speed"):
        assemble(work_folder.dir, out, speed=speed, meta=META)

    assert not out.exists()
    assert not (tmp_path / "book.chapters").exists()


def test_wild_speed_warns_once_and_still_produces_audio(tmp_path, work_folder):
    warnings = Warnings()

    out = assemble(work_folder.dir, tmp_path / "book.m4b", speed=1.6, meta=META, events=warnings)

    assert out.is_file()
    assert len(warnings.messages) == 1
    assert "speed" in warnings.messages[0].lower()


def test_partial_render_warns_and_assembles_the_finished_chapter(tmp_path, work_folder_factory):
    partial = work_folder_factory(["Opening", "The Long Road"], rendered=1)
    warnings = Warnings()

    out = assemble(partial.dir, tmp_path / "book.m4b", speed=1.0, meta=META, events=warnings)

    assert out.is_file()
    assert any("not fully rendered" in m for m in warnings.messages)
    chapters_dir = out.parent / (out.stem + ".chapters") / "speed-1"
    assert len(list(chapters_dir.glob("*.flac"))) == 1


def test_nothing_rendered_is_an_error_not_an_empty_book(tmp_path, work_folder_factory):
    empty = work_folder_factory(["Opening", "The Long Road"], rendered=0)

    with pytest.raises(ValueError, match="no Chapter"):
        assemble(empty.dir, tmp_path / "book.m4b", speed=1.0, meta=META)


def test_changing_speed_never_touches_the_takes(tmp_path, work_folder):
    takes_before = _snapshot(work_folder.takes_dir)

    assemble(work_folder.dir, tmp_path / "a.m4b", speed=1.15, meta=META)
    assemble(work_folder.dir, tmp_path / "b.m4b", speed=0.85, meta=META)

    assert _snapshot(work_folder.takes_dir) == takes_before
    assert os.path.exists(work_folder.takes_dir)

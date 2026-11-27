"""Seam 2: Assembly over a finished work folder. Reading speed is applied to Chapter audio, never to Takes."""

from __future__ import annotations

import hashlib

import pytest
import soundfile as sf

from folivox_ming.assemble import stretch_chapters


def _duration(path) -> float:
    return sf.info(str(path)).duration


def _digests(folder) -> dict[str, str]:
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("*")) if p.is_file()}


def test_returns_chapters_in_order_with_measured_durations(tmp_path, work_folder):
    out = tmp_path / "out"

    built = stretch_chapters(work_folder.dir, out, speed=1.0)

    assert [c.title for c in built] == work_folder.chapters
    for chapter in built:
        assert chapter.path.is_file()
        assert chapter.duration == pytest.approx(_duration(chapter.path), abs=0.02)


def test_speed_changes_duration_of_the_written_files(tmp_path, work_folder):
    faster = stretch_chapters(work_folder.dir, tmp_path / "fast", speed=1.15)
    slower = stretch_chapters(work_folder.dir, tmp_path / "slow", speed=0.85)
    plain = stretch_chapters(work_folder.dir, tmp_path / "plain", speed=1.0)

    assert faster[0].duration == pytest.approx(plain[0].duration / 1.15, rel=0.02)
    assert slower[0].duration == pytest.approx(plain[0].duration / 0.85, rel=0.02)
    assert faster[0].duration == pytest.approx(_duration(faster[0].path), abs=0.02)


def test_takes_and_folivox_chapter_files_are_left_alone(tmp_path, work_folder):
    takes_before = _digests(work_folder.takes_dir)
    chapters_before = _digests(work_folder.chapters_dir)

    stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.15)

    assert _digests(work_folder.takes_dir) == takes_before
    assert _digests(work_folder.chapters_dir) == chapters_before


def test_each_speed_gets_its_own_folder_and_nothing_is_overwritten(tmp_path, work_folder):
    plain = stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.0)
    stretched = stretch_chapters(work_folder.dir, tmp_path / "out", speed=1.15)

    assert plain[0].path != stretched[0].path
    assert plain[0].path.is_file() and stretched[0].path.is_file()
    assert plain[0].path.parent != stretched[0].path.parent

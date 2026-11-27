"""Assembly: reading speed applied to Chapter audio, without touching the Takes.

Folivox owns the render; this module owns what happens after it. A finished work folder already holds
every Take and usually every Chapter file, so pacing a book is a stretch of the chapter audio — seconds of
work rather than the hours a re-render would cost (see ADR 0002).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import soundfile as sf
from folivox.audio import audio_seconds
from folivox.audio.assemble import BookMeta, ChapterAudio, build_chapter, run
from folivox.journal import Journal


def speed_folder(out_dir: Path, speed: float) -> Path:
    """Where one reading speed's Chapter files live: a folder per speed, so none overwrites another."""
    return Path(out_dir) / f"speed-{speed:g}"


def chapter_source(journal: Journal, chapter: int, work_dir: Path, scratch: Path) -> Path:
    """The Chapter file to stretch: Folivox's own, or one built from the same Takes if the render never got there."""
    existing = work_dir / "chapters" / f"{chapter + 1:03d}.flac"
    if existing.is_file():
        return existing
    parts = [(journal.audio_path(c.key), c.pause) for c in journal.chunks(chapter)]
    built = scratch / f"{chapter + 1:03d}.flac"
    built.parent.mkdir(parents=True, exist_ok=True)
    build_chapter(parts, built, _sample_rate(journal, parts))
    return built


def _sample_rate(journal: Journal, parts: list[tuple[Path, float]]) -> int:
    rate = journal.meta("sample_rate")
    if rate:
        return int(rate)
    return int(sf.info(str(parts[0][0])).samplerate)


def stretch_file(src: Path, dst: Path, speed: float, sample_rate: int) -> float:
    """Write ``src`` to ``dst`` at ``speed``, and return the duration of the file as written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if speed == 1.0:
        shutil.copy2(src, dst)
    else:
        run(["-i", str(src), "-af", f"atempo={speed}", "-ar", str(sample_rate), "-ac", "1",
             "-c:a", "flac", str(dst)])
    return audio_seconds(dst) or sf.info(str(dst)).duration


def stretch_chapters(work_dir: Path, out_dir: Path, speed: float, events=None) -> list[ChapterAudio]:
    """Every fully rendered Chapter, stretched to ``speed``, in chapter order.

    Chapters whose render is incomplete are skipped with a warning rather than assembled from half a chapter.
    """
    work_dir = Path(work_dir)
    target = speed_folder(out_dir, speed)
    scratch = Path(out_dir) / "_built"
    journal = Journal(work_dir)
    built: list[ChapterAudio] = []
    try:
        for chapter, title in journal.chapters():
            chunks = journal.chunks(chapter)
            if not chunks or not all(journal.is_done(c.key) for c in chunks):
                _warn(events, f"Chapter “{title}” is not fully rendered — skipping")
                continue
            src = chapter_source(journal, chapter, work_dir, scratch)
            rate = _sample_rate(journal, [(src, 0.0)])
            dst = target / f"{chapter + 1:03d}.flac"
            duration = stretch_file(src, dst, speed, rate)
            built.append(ChapterAudio(title, dst, duration))
    finally:
        journal.close()
    return built


def _warn(events, message: str) -> None:
    if events is not None and hasattr(events, "warning"):
        events.warning(message)

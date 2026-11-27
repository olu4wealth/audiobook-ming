"""Assembly: reading speed applied to Chapter audio, without touching the Takes.

Folivox owns the render; this module owns what happens after it. A finished work folder already holds
every Take and usually every Chapter file, so pacing a book is a stretch of the chapter audio — seconds of
work rather than the hours a re-render would cost (see ADR 0002).
"""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

import soundfile as sf
from folivox.audio import audio_seconds
from folivox.audio.assemble import BookMeta, ChapterAudio, build_chapter, run, write_m4b
from folivox.journal import Journal

SANE_SPEED = (0.85, 1.15)


def folder_for(out: Path, speed: float) -> Path:
    """Where a book's Chapter files live: ``<book>.chapters/speed-<n>``, one folder per reading speed."""
    out = Path(out)
    return out.parent / f"{out.stem}.chapters" / f"speed-{speed:g}"


def assemble(work_dir: Path, out: Path, speed: float = 1.0, meta: BookMeta | None = None,
             cover: bytes | None = None, events=None, bitrate: str = "64k") -> Path:
    """Stretch every finished Chapter to ``speed`` and write the chaptered M4B beside ``out``."""
    check_speed(speed, events)
    chapters = stretch_chapters(work_dir, Path(out).parent, speed, events, book_stem=Path(out).stem)
    return write_audiobook(chapters, out, meta or journal_meta(work_dir), cover, bitrate)


def journal_meta(work_dir: Path) -> BookMeta:
    """Metadata for a work folder assembled on its own: only what the render actually recorded."""
    journal = Journal(Path(work_dir))
    try:
        title = journal.meta("book_title") or Path(work_dir).name
    finally:
        journal.close()
    return BookMeta(title=title)


def stretch_chapters(work_dir: Path, out_parent: Path, speed: float, events=None,
                     book_stem: str = "") -> list[ChapterAudio]:
    """Every fully rendered Chapter, stretched to ``speed``, in chapter order.

    Chapters whose render is incomplete are skipped with a warning rather than assembled from half a chapter.
    A chapter already stretched to this speed is reused as it stands.
    """
    work_dir = Path(work_dir)
    target = _target_folder(out_parent, book_stem, speed)
    scratch = Path(out_parent) / "_built"
    journal = Journal(work_dir)
    built: list[ChapterAudio] = []
    try:
        for chapter, title in journal.chapters():
            chunks = journal.chunks(chapter)
            if not chunks or not all(journal.is_done(c.key) for c in chunks):
                _warn(events, f"Chapter “{title}” is not fully rendered — skipping")
                continue
            dst = target / f"{chapter + 1:03d}.flac"
            signature = _signature(chunks)
            duration = _cached(journal, chapter, signature, speed, dst)
            if duration is None:
                src = chapter_source(journal, chapter, work_dir, scratch)
                rate = _sample_rate(journal, [(src, 0.0)])
                duration = stretch_file(src, dst, speed, rate)
                journal.set_meta(**{_cache_key(chapter): {"sig": signature, "speed": speed,
                                                          "size": dst.stat().st_size}})
            built.append(ChapterAudio(title, dst, duration))
    finally:
        journal.close()
    return built


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


def stretch_file(src: Path, dst: Path, speed: float, sample_rate: int) -> float:
    """Write ``src`` to ``dst`` at ``speed``, and return the duration of the file as written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if speed == 1.0:
        shutil.copy2(src, dst)
    else:
        run(["-i", str(src), "-af", f"atempo={speed}", "-ar", str(sample_rate), "-ac", "1",
             "-c:a", "flac", str(dst)])
    return audio_seconds(dst) or sf.info(str(dst)).duration


def write_audiobook(chapters: list[ChapterAudio], out: Path, meta: BookMeta, cover: bytes | None = None,
                    bitrate: str = "64k") -> Path:
    """The chaptered M4B, with marks taken from each Chapter's post-stretch duration.

    Folivox writes the file next to the target and renames it, so a failed encode cannot damage an earlier
    good book.
    """
    if not chapters:
        raise ValueError("Nothing to assemble: no Chapter is fully rendered")
    out = Path(out)
    write_m4b(chapters, out, meta, cover, bitrate)
    return out


def _target_folder(out_parent: Path, book_stem: str, speed: float) -> Path:
    if book_stem:
        return folder_for(Path(out_parent) / f"{book_stem}.m4b", speed)
    return Path(out_parent) / f"speed-{speed:g}"


def check_speed(speed: float, events=None) -> None:
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed != speed \
            or math.isinf(speed) or speed <= 0:
        raise ValueError(f"reading speed must be a positive number, got {speed!r}")
    if not SANE_SPEED[0] <= speed <= SANE_SPEED[1]:
        _warn(events, f"Reading speed {speed:g} is outside {SANE_SPEED[0]:g}–{SANE_SPEED[1]:g}: "
                      f"the artefacts of stretching become audible")


def _cached(journal: Journal, chapter: int, signature: str, speed: float, dst: Path) -> float | None:
    """The duration of an unchanged chapter already stretched to this speed, or None to stretch it now."""
    record = journal.meta(_cache_key(chapter))
    if not isinstance(record, dict) or record.get("sig") != signature or record.get("speed") != speed:
        return None
    try:
        if dst.stat().st_size != record.get("size"):
            return None
    except OSError:
        return None
    return audio_seconds(dst) or sf.info(str(dst)).duration


def _cache_key(chapter: int) -> str:
    return f"ming_assembly_{chapter}"


def _signature(chunks) -> str:
    """The same chapter identity Folivox uses for its own cache: the Takes and their pauses."""
    return hashlib.sha1("|".join(f"{c.key}:{c.pause}" for c in chunks).encode()).hexdigest()


def _sample_rate(journal: Journal, parts: list[tuple[Path, float]]) -> int:
    rate = journal.meta("sample_rate")
    if rate:
        return int(rate)
    return int(sf.info(str(parts[0][0])).samplerate)


def _warn(events, message: str) -> None:
    if events is not None and hasattr(events, "warning"):
        events.warning(message)

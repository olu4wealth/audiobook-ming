"""Fixtures that build a finished Folivox work folder the way a real render leaves one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from folivox.journal import Journal, Take
from folivox.text.chunk import Chunk

SAMPLE_RATE = 24000
TONE_SECONDS = 1.0


@dataclass
class WorkFolder:
    dir: Path
    chapters: list[str]
    sample_rate: int

    @property
    def takes_dir(self) -> Path:
        return self.dir / "audio"

    @property
    def chapters_dir(self) -> Path:
        return self.dir / "chapters"


def _tone(seconds: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def build_work_folder(root: Path, chapters: list[str], *, rendered: int | None = None,
                      tone_seconds: float = TONE_SECONDS) -> WorkFolder:
    """A journal whose chapters each hold one rendered Take, plus optional pre-built chapter FLACs.

    ``rendered`` limits how many chapters have Takes (the rest are planned but never synthesised).
    """
    work = root / "work"
    journal = Journal(work)
    plan = [(title, [Chunk(index=0, text=f"Chapter {i}.", source=f"Chapter {i}.", pause_after=0.2)])
            for i, title in enumerate(chapters)]
    journal.set_plan(plan, voice_id="russlib", settings="ming|speed=1.0")
    journal.set_meta(sample_rate=SAMPLE_RATE, book_title="Fixture Book")

    limit = len(chapters) if rendered is None else rendered
    chapters_dir = work / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    for chunk, (ci, _title) in zip(journal.chunks(), journal.chapters()):
        if ci >= limit:
            continue
        sf.write(str(journal.audio_path(chunk.key)), _tone(tone_seconds, SAMPLE_RATE), SAMPLE_RATE,
                 subtype="PCM_16")
        journal.record(Take(key=chunk.key, duration=tone_seconds, score=None, asr=None, takes=1,
                            synth_seconds=tone_seconds))
        chapter_path = chapters_dir / f"{ci + 1:03d}.flac"
        sf.write(str(chapter_path), _tone(tone_seconds + 0.5, SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")
    journal.close()
    return WorkFolder(dir=work, chapters=chapters, sample_rate=SAMPLE_RATE)


@pytest.fixture
def work_folder_factory(tmp_path):
    def make(chapters: list[str] = ("Opening", "The Long Road"), **kwargs) -> WorkFolder:
        return build_work_folder(tmp_path, list(chapters), **kwargs)

    return make


@pytest.fixture
def work_folder(work_folder_factory) -> WorkFolder:
    return work_folder_factory()

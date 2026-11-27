"""Seam 1: the engine drives a worker process over JSON lines, with no GPU and no checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from folivox.tts.base import SynthTimeout

from folivox_ming import install
from folivox_ming.engine import MingEngine, MingError, language_for

STUB_ENV = Path(__file__).parent / "fixtures" / "stub_env"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(STUB_ENV))
    monkeypatch.setattr(install, "venv_python", lambda: Path(sys.executable))
    eng = MingEngine(model_dir=str(weights), repo_dir=str(repo), device="cpu")
    yield eng
    eng.close()


@pytest.fixture
def voice(tmp_path):
    reference = tmp_path / "narrator.wav"
    sf.write(str(reference), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
    return SimpleNamespace(id="narrator", wav=reference, transcript="Hello there.", warnings=[])


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_MING_LOG", str(path))

    def read() -> list[dict]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    return read


def test_start_learns_the_models_real_sample_rate(engine, voice):
    engine.start(voice)

    assert engine.sample_rate == 24000
    assert engine.load_seconds >= 0


def test_synthesis_returns_mono_float32_at_the_render_rate(engine, voice):
    audio, seconds = engine.synthesize_audio("Hello there.", voice, seed=1, speed=1.0)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(0.25 * 24000, rel=0.02)
    assert seconds >= 0


def test_text_normalisation_is_switched_off_when_pynini_is_absent(engine, voice, log):
    """Ming's FST normalizer needs pynini, which has no universal wheel.

    Constructing the model with normalization on and no pynini raises during warm-up, after the checkpoint has
    already loaded — which is exactly how the GPU smoke test failed. Ming documents turning it off instead.
    """
    engine.start(voice)

    construct = [entry for entry in log() if entry.get("construct")]
    assert construct, "the stub never saw the model being constructed"
    assert construct[-1]["text_normalization"] is False


@pytest.mark.parametrize(("text", "task", "language_id"), [
    ("Hello there.", "TTS", "English"),
    ("你好", "TTS", "Chinese"),
    ("こんにちは", "MULTILINGUAL", "Japanese"),
    ("안녕하세요", "MULTILINGUAL", "Korean"),
    ("Bonjour tout le monde.", "TTS", "English"),  # Latin text is read as English: documented limit
])
def test_each_fragment_is_read_in_the_right_task(engine, voice, log, text, task, language_id):
    engine.synthesize_audio(text, voice, seed=1, speed=1.0)

    call = log()[-1]
    assert (call["task"], call["language_id"]) == (task, language_id)
    assert call["use_spk_emb"] is True


def test_language_for_ignores_latin_and_prefers_kana_over_kanji():
    assert language_for("日本語のテキスト") == "ja"
    assert language_for("plain english") == "en"
    assert language_for("漢字") == "zh"  # kanji alone, no kana
    assert language_for("한국어") == "ko"


def test_a_failed_fragment_does_not_kill_the_worker(engine, voice):
    with pytest.raises(MingError, match="stub was asked to fail"):
        engine.synthesize_audio("BOOM", voice, seed=1, speed=1.0)

    audio, _ = engine.synthesize_audio("Hello again.", voice, seed=1, speed=1.0)
    assert len(audio) > 0


def test_a_killed_worker_is_reported_as_a_memory_hint(engine, voice):
    with pytest.raises(MingError) as excinfo:
        engine.synthesize_audio("DIE", voice, seed=1, speed=1.0)

    message = str(excinfo.value)
    assert "exit code 137" in message
    assert "memory" in message.lower()


def test_a_silent_worker_times_out_with_a_resume_hint(engine, voice, monkeypatch):
    monkeypatch.setenv("FOLIVOX_SYNTH_TIMEOUT", "2")

    with pytest.raises(SynthTimeout) as excinfo:
        engine.synthesize_audio("HANG", voice, seed=1, speed=1.0)

    assert "run the same command again" in str(excinfo.value).lower()


def test_close_leaves_no_child_process(engine, voice):
    engine.start(voice)
    worker = engine._worker
    assert worker is not None and worker.proc.poll() is None

    engine.close()

    assert engine._worker is None
    assert worker.proc.poll() is not None

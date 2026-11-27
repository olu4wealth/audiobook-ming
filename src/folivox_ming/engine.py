"""The Ming engine: a Folivox ``TTSEngine`` backed by a worker in Ming's own environment.

Folivox never imports Ming here either. ``worker.py`` runs under the interpreter of the Engine environment
and talks JSON lines over pipes; a native crash or an out-of-memory kill takes down only the worker, and
closing it gives the GPU memory back.
"""

from __future__ import annotations

import collections
import json
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from folivox import memory
from folivox.tts.base import SynthTimeout, TTSEngine, synth_timeout
from folivox.tts.voice import Voice
from folivox_ming import install

WORKER = Path(__file__).with_name("worker.py")
LOAD_TIMEOUT = 1800.0  # reading the checkpoint and the kernel warm-up

# The cloned-voice path is ``TTS`` for the two languages its task list names, ``MULTILINGUAL`` for the rest.
MING_TASKS = {
    "zh": ("TTS", "Chinese"),
    "en": ("TTS", "English"),
    "ja": ("MULTILINGUAL", "Japanese"),
    "ko": ("MULTILINGUAL", "Korean"),
    "de": ("MULTILINGUAL", "German"),
    "es": ("MULTILINGUAL", "Spanish"),
    "fr": ("MULTILINGUAL", "French"),
}



class MingError(RuntimeError):
    pass


def language_for(text: str) -> str:
    """The language of one fragment: Folivox splits fragments by language already, so the script decides it.

    Kana means Japanese even when kanji outnumber it; hangul means Korean; Han alone means Chinese. Latin
    text is read as English — Ming has no detection of its own, and a fragment cannot say it is French.
    """
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:  # hiragana, katakana
            return "ja"
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:  # hangul syllables, jamo
            return "ko"
    if any(0x4E00 <= ord(c) <= 0x9FFF for c in text):
        return "zh"
    return "en"


def to_reference_wav(src: Path, dst: Path) -> Path:
    """Mono 44.1 kHz WAV of a reference voice, whatever format the user (or a bundled clip) supplied."""
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "44100",
                        str(dst)], capture_output=True, text=True)  # noqa: S603, S607
    if r.returncode != 0:
        dst.unlink(missing_ok=True)  # a partial file would be taken for a finished one next time
        raise MingError(f"Could not prepare the reference voice {src.name}: {r.stderr.strip()[-300:]}")
    return dst


class _Worker:
    """The child process and its two pipes; replies are read by a thread so a hang can time out."""

    def __init__(self, python: Path):
        self.proc = subprocess.Popen([str(python), str(WORKER)], stdin=subprocess.PIPE,  # noqa: S603
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     encoding="utf-8", bufsize=1)
        self.replies: queue.Queue = queue.Queue()
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        threading.Thread(target=self._read_out, daemon=True, name="folivox-ming-out").start()
        threading.Thread(target=self._read_err, daemon=True, name="folivox-ming-err").start()

    def _read_out(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            try:
                self.replies.put(json.loads(line))
            except ValueError:
                self.stderr_tail.append(line.rstrip())
        self.replies.put(None)  # EOF: the worker is gone

    def _read_err(self) -> None:
        for line in self.proc.stderr:  # type: ignore[union-attr]
            self.stderr_tail.append(line.rstrip())

    def call(self, req: dict, timeout: float) -> dict:
        if self.proc.poll() is not None:
            raise MingError(self._died())
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except OSError as e:
            raise MingError(self._died()) from e
        try:
            rep = self.replies.get(timeout=timeout or None)
        except queue.Empty:
            self.kill()
            raise SynthTimeout(f"The Ming worker did not answer in {timeout:.0f} s and was stopped. Progress is "
                               "saved: run the same command again to continue (set FOLIVOX_SYNTH_TIMEOUT=seconds "
                               "to change the limit, 0 to turn it off).") from None
        if rep is None:
            raise MingError(self._died())
        if not rep.get("ok"):
            raise MingError(rep.get("error", "unknown error"))
        return rep

    def _died(self) -> str:
        try:
            code = self.proc.wait(timeout=2)  # the pipe closes a moment before Windows reports the exit code
        except subprocess.TimeoutExpired:
            code = self.proc.poll()
        tail = "\n".join(list(self.stderr_tail)[-8:])
        hint = " It was probably killed for lack of memory: close heavy applications and try again." \
            if code in (-9, 137) else ""
        return f"The Ming worker stopped (exit code {code}).{hint}\n{tail}".rstrip()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)  # reap it: the GPU memory is released only when the process is gone
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"cmd": "quit"}\n')  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.kill()
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass


class MingEngine(TTSEngine):
    """Ming-3.0-tts-0.5b: zero-shot cloning, 7 languages, one worker process."""

    name = "ming"
    device = "cuda"
    languages = tuple(MING_TASKS)
    local = True

    def __init__(self, model_dir: str | None = None, device: str = "auto", repo_dir: str | None = None,
                 **_ignored: Any):
        self.model_dir = Path(model_dir).expanduser() if model_dir else install.weights_dir()
        self.repo_dir = Path(repo_dir).expanduser() if repo_dir else install.repo_dir()
        python = install.venv_python()
        if not python.is_file() or not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"{install.TITLE} is not installed. Install it first (it downloads "
                                    f"~3 GB and builds a private environment): python -m folivox_ming.install")
        self.device = "cpu" if device == "cpu" else "cuda"
        # Folivox's warm-up message reads ``engine.build.title``; the shim keeps that message honest.
        self.build = SimpleNamespace(title=install.TITLE, id=install.ENGINE_ID, revision=install.REF)
        self._python = python
        self._worker: _Worker | None = None
        self._refs: dict[str, Path] = {}
        self._tmp = tempfile.TemporaryDirectory(prefix="folivox-ming-engine-")
        self.peak_gb = 0.0
        self.load_seconds = 0.0
        self._calls = 0

    # -- life cycle
    def start(self, voice: Voice) -> None:
        """Load the model and warm it up with the book's own voice: the first call compiles kernels, and the
        warm-up is the only way to learn the model's real sample rate, which the journal is told before any take."""
        if self._worker is not None:
            return
        ref = self._reference(voice)
        self._worker = _Worker(self._python)
        t0 = time.time()
        try:
            rep = self._worker.call({"cmd": "load", "model_dir": str(self.model_dir),
                                     "repo_dir": str(self.repo_dir), "device": self.device,
                                     "warmup": {"wav": str(ref), "text": voice.transcript}}, LOAD_TIMEOUT)
        except BaseException:
            # a failed load must not leave a model-less worker behind: the next start() would skip loading
            self._worker.close()
            self._worker = None
            raise
        if rep.get("sample_rate"):
            self.sample_rate = int(rep["sample_rate"])
        self.load_seconds = time.time() - t0

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._tmp.cleanup()
        memory.release()

    def __del__(self):  # pragma: no cover — a safety net; Folivox closes the engine explicitly
        try:
            if self._worker is not None:
                self._worker.close()
        except Exception:  # noqa: BLE001
            pass

    # -- synthesis
    def _reference(self, voice: Voice) -> Path:
        if voice.id not in self._refs:
            self._refs[voice.id] = to_reference_wav(voice.wav, Path(self._tmp.name) / f"ref-{voice.id}.wav")
        return self._refs[voice.id]

    def synthesize_audio(self, text: str, voice: Voice, seed: int, speed: float) -> tuple[np.ndarray, float]:
        # Reading speed is not a parameter of this model: it is applied to the assembled audio instead
        # (docs/adr/0002-speed-applied-at-assembly.md). Folivox warns once about the unused setting.
        self.start(voice)
        assert self._worker is not None
        task, language_id = MING_TASKS[language_for(text)]
        self._calls += 1
        out = Path(self._tmp.name) / f"out-{self._calls}.npy"
        rep = self._worker.call({"cmd": "synth", "id": self._calls, "text": text,
                                 "ref": {"wav": str(self._reference(voice)), "text": voice.transcript},
                                 "task": task, "language_id": language_id, "seed": seed,
                                 "sample_rate": self.sample_rate, "out": str(out)},
                                synth_timeout(len(text)))
        audio = np.load(out).astype(np.float32)
        out.unlink(missing_ok=True)
        self.peak_gb = max(self.peak_gb, float(rep.get("peak_gb", 0.0)))
        return audio, float(rep["seconds"])

"""Load the Ming model on CPU and synthesise two short fragments.

Not a supported path — the model wants a GPU — but it answers three questions no test can: does the real
``MingTTS`` accept the arguments the worker passes, what sample rate does it answer in, and how long does a
take actually need. Watchdog-killed so a hung load cannot outlive the session.
"""

from __future__ import annotations

import os
import threading
import time

DEADLINE = float(os.environ.get("PROBE_DEADLINE", "900"))


def _watchdog() -> None:
    print(f"WATCHDOG: giving up after {DEADLINE:.0f}s", flush=True)
    os._exit(9)


def main() -> None:
    from folivox import voices
    from folivox.tts.voice import Voice

    from folivox_ming.engine import MingEngine

    threading.Thread(target=lambda: (time.sleep(DEADLINE), _watchdog()), daemon=True).start()

    entry = voices.find("en-kalynda")
    voice = Voice(entry.path, entry.transcript, 0.0)
    engine = MingEngine(device="cpu")
    started = time.time()
    engine.start(voice)
    print(f"LOAD_SECONDS {time.time() - started:.1f} SAMPLE_RATE {engine.sample_rate}", flush=True)

    for text in ("Hello there.", "こんにちは。"):
        t0 = time.time()
        audio, seconds = engine.synthesize_audio(text, voice, 0, 1.0)
        print(f"TAKE {text!r} samples={len(audio)} dtype={audio.dtype} "
              f"audio_s={len(audio) / engine.sample_rate:.2f} wall_s={time.time() - t0:.1f} "
              f"reported_s={seconds:.2f} peak_gb={engine.peak_gb:.2f}", flush=True)

    engine.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

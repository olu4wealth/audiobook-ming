"""Ming worker: the process that holds the model, running the Engine environment's Python.

Standalone by design — it must not import Folivox, and Folivox never imports Ming code. Protocol: one JSON
object per line on stdin, one JSON reply per line on the real stdout (everything torch prints goes to stderr).

    {"cmd": "load", "model_dir": dir, "repo_dir": dir, "device": "cuda" | "cpu",
     "warmup": {"wav": path, "text": str} | null}
        → {"ok": true, "sample_rate": 24000, "load_s": .., "warm_s": ..}
    {"cmd": "synth", "id": n, "text": str, "ref": {"wav": path, "text": str},
     "task": "TTS" | "MULTILINGUAL", "language_id": "English", "seed": int,
     "sample_rate": 44100 | null, "out": "x.npy"}
        → {"ok": true, "id": n, "out": "x.npy", "sample_rate": 24000, "audio_s": .., "seconds": .., "peak_gb": ..}
    {"cmd": "quit"}

``sample_rate`` in a synth request is the rate the rest of the render is already using; the worker resamples
when the model answers with another one (a different task may), so one book never mixes rates.

Any failure replies {"ok": false, "error": "..."}; the process stays alive for the next request.
"""

import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

SCRATCH = tempfile.mkdtemp(prefix="folivox-ming-")
STATE: dict = {}


def main() -> None:
    # Requests arrive as UTF-8 JSON: the engine writes UTF-8 whatever the console code page is, and stdin's
    # default (the locale, with surrogateescape) would turn CJK text into lone surrogates before synthesis.
    sys.stdin.reconfigure(encoding="utf-8")
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)  # stray prints from the libraries must not corrupt the protocol channel
    sys.stdout = sys.stderr

    def reply(obj: dict) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req["cmd"]
            if cmd == "quit":
                return
            if cmd == "load":
                t0 = time.time()
                if req.get("device") == "cpu":
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported
                sys.path.insert(0, req["repo_dir"])
                import soundfile  # noqa: F401 — fail here, not inside the first synthesis

                from ming_tts import MingTTS

                STATE["model"] = MingTTS(req["model_dir"], text_normalization=_has_pynini())
                load_s = time.time() - t0
                sample_rate, warm_s = 0, 0.0
                if req.get("warmup"):
                    # the first generation compiles kernels: pay it now, and learn the model's real rate
                    t1 = time.time()
                    w = req["warmup"]
                    _audio, sample_rate = generate("Hello there.", w["wav"], w["text"], "TTS", "English", 0)
                    warm_s = time.time() - t1
                STATE["sample_rate"] = sample_rate
                reply({"ok": True, "sample_rate": sample_rate, "load_s": load_s, "warm_s": warm_s})
            elif cmd == "synth":
                t0 = time.time()
                audio, sr = generate(req["text"], req["ref"]["wav"], req["ref"]["text"], req["task"],
                                     req["language_id"], req.get("seed", 0), req.get("sample_rate"))
                np.save(req["out"], audio)
                reply({"ok": True, "id": req.get("id"), "out": req["out"], "sample_rate": sr,
                       "audio_s": len(audio) / sr, "seconds": time.time() - t0, "peak_gb": peak_gb()})
            else:
                reply({"ok": False, "error": f"unknown command {cmd!r}"})
        except Exception as e:  # noqa: BLE001 — report and keep serving
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _has_pynini() -> bool:
    """Whether Ming's FST text normalizer can be switched on.

    It needs ``pynini``, which has no universal wheel (Ming's README says to prefer conda). Without it the model
    still synthesizes; it just reads numbers, dates and units as written. Turning normalization off explicitly
    is what Ming documents, and is the difference between a render and a hard failure on a host without it.
    """
    try:
        import pynini  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not usable here"
        return False
    return True


def generate(text: str, ref_wav: str, ref_text: str, task: str, language_id: str, seed: int,
             target_rate: int | None = None):
    """One take: clone ``ref_wav`` (transcript ``ref_text``) and read ``text`` in it."""
    import soundfile as sf
    import torch

    model = STATE["model"]
    try:
        torch.manual_seed(int(seed))
    except Exception:  # noqa: BLE001
        pass
    out = os.path.join(SCRATCH, f"take-{time.time_ns()}.wav")
    try:
        model.synthesize(task=task, language_id=language_id, text=text, prompt_wav_path=ref_wav,
                         prompt_text=ref_text, use_spk_emb=True, output_path=out)
        audio, sr = sf.read(out, dtype="float32", always_2d=False)
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_rate and int(target_rate) != int(sr):
        audio, sr = resample(audio, int(sr), int(target_rate)), int(target_rate)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def resample(audio, src: int, dst: int):
    """Linear resample: a guard for a model answering in another rate than the render's, not a quality path."""
    n = max(1, int(round(len(audio) * dst / src)))
    x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    y = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(y, x, audio).astype(np.float32)


def peak_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        pass
    return 0.0


if __name__ == "__main__":
    main()

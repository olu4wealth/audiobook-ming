"""Stand-in for Ming's model, imported by name inside the worker process during tests.

It speaks the real ``MingTTS(model_dir).synthesize(...)`` signature and writes a real WAV, so the worker's
own code path (soundfile read, mono downmix, resample) is exercised without a GPU or a checkpoint. Failures
are triggered by words in the text, so one worker can be driven through every path.
"""

import json
import os
import time

import numpy as np
import soundfile as sf

SAMPLE_RATE = int(os.environ.get("STUB_MING_SR", "24000"))


def _log(entry: dict) -> None:
    path = os.environ.get("STUB_MING_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class MingTTS:
    def __init__(self, model_dir, **_ignored):
        self.model_dir = model_dir

    def synthesize(self, *, task, language_id, text, prompt_wav_path, prompt_text, use_spk_emb=True,
                   output_path):
        _log({"task": task, "language_id": language_id, "text": text, "use_spk_emb": use_spk_emb,
              "ref": prompt_wav_path, "ref_text": prompt_text})
        if "BOOM" in text:
            raise RuntimeError("the stub was asked to fail")
        if "DIE" in text:
            os._exit(137)  # like an OOM kill
        if "HANG" in text:
            time.sleep(600)
        seconds = 0.25
        t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
        sf.write(output_path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SAMPLE_RATE,
                 subtype="PCM_16")

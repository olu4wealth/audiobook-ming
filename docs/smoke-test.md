# GPU smoke test

The one thing no local test can settle: that Ming loads, what it actually costs, and what sample rate it
really answers in. One chapter, one bundled voice, about ten minutes of setup.

Everything here runs on a Kaggle notebook (free 16 GB GPU). Nothing else in the project needs it: the test
suite runs without a GPU, and the installer's clone/environment/checkpoint steps are already verified on CPU.

## Before you start

In the notebook's right-hand panel:

1. **Accelerator**: GPU T4 ×1 (16 GB is enough for the 0.5B checkpoint).
2. **Internet**: on.
3. **Add-ons → Secrets**: a secret named `GH_TOKEN` holding a GitHub token with `repo` scope — this
   repository is private, and the package installs from it.

Kaggle's working disk is `/kaggle/working` (~20 GB); everything below lives there so it survives the session.

## 1. Install (≈10–20 min; torch and ~3 GB of weights dominate)

```bash
pip install -q "git+https://github.com/vatavale/folivox"
pip install -q "git+https://x-access-token:${GH_TOKEN}@github.com/olu4wealth/audiobook-ming"

export FOLIVOX_MING_HOME=/kaggle/working/engines
python -m folivox_ming.install
python -m folivox_ming.install --check      # must end "is installed (...)"
folivox providers                           # `ming` must be listed
```

`FOLIVOX_MING_HOME` keeps the Engine environment and the checkpoint on the working disk instead of the home
directory. The installer uses the running interpreter (Kaggle's Python is 3.11, which Ming supports) because
`uv` is not installed; if it ever is, the installer prefers it.

## 2. Make the smallest possible Book

```python
from pathlib import Path
Path("/kaggle/working/book.txt").write_text(
    "Chapter One\n\n"
    "The lighthouse stood at the end of the spit, and every evening it counted the boats home. "
    "On the night the storm came up, the keeper climbed the stairs twice as fast as he had in years.\n",
    encoding="utf-8")
```

One chapter, two sentences: enough to exercise load, warm-up, a handful of Takes and the assembly, and short
enough to re-run when something fails.

## 3. Render

```bash
python -m folivox_ming render /kaggle/working/book.txt \
    --out /kaggle/working/book.m4b \
    --voice en-kalynda \
    --speed 1.0
```

Expect: a progress line per stage on stderr, `book.chapters/speed-1/001.flac`, then `book.m4b`.

## 4. Record what the run proves

```bash
python - <<'PY'
import json, sqlite3
db = sqlite3.connect("/kaggle/working/book.folivox/journal.sqlite")
meta = dict(db.execute("SELECT key, value FROM meta"))
for key in ("sample_rate", "settings", "book_title"):
    print(key, "=", meta.get(key))
print("takes:", db.execute("SELECT count(*), round(sum(duration), 2) FROM takes").fetchone())
PY

ffprobe -v error -show_entries format=duration -of default=nw=1 /kaggle/working/book.m4b
ffprobe -v error -print_format json -show_chapters /kaggle/working/book.m4b | head -40
free -g; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
```

Paste back:

- **`sample_rate`** — the value the warm-up learned. If it is not 24000, the engine's `MING_TASKS` and the
  resample guard matter more than expected; nothing needs changing, but note it.
- **`takes`** — how many Fragments the chapter became, and their total audio seconds.
- **Wall clock** — time from `render` to the M4B, and the load portion from the progress lines.
- **VRAM used during the run** — the engine prints a peak per take; compare with the 16 GB budget.
- **Whether the M4B's one chapter mark has the right title and duration** — the assembly's stretch path was
  exercised on CPU only, with a 1.0 speed here.

## What to do with a failure

| Symptom | Most likely cause |
|---|---|
| `MingTTS` import error from the worker | `requirements.txt` changed upstream; re-run the installer (it continues) |
| `The Ming worker stopped (exit code 137)` | Out of memory — a bigger card, or shorten the chapter |
| `did not answer in N s` | Same fragment, stuck: the run resumes; raise `FOLIVOX_SYNTH_TIMEOUT` |
| Garbled or empty audio | The reference clip: try another `--voice` id from `folivox voices` |
| Chapter files but no M4B | ffmpeg on the host; `ffmpeg -version` in the notebook |

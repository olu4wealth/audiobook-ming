# Spec 0001 — Audiobooks on Ming-3.0-tts

Status: ready for an agent. Tracker: not published (no issue tracker configured; this file is the record).

Vocabulary is the project's: [GLOSSARY](../../GLOSSARY.md). Decisions live in
[ADR 0001](../adr/0001-ming-as-out-of-tree-provider.md) and
[ADR 0002](../adr/0002-speed-applied-at-assembly.md).

## Problem Statement

I have books — EPUB, PDF, plain text — and I want them as audiobooks read in a voice I chose, with chapter
markers, on my own machine, without paying a per-character API and without a multi-hour run being lost the
first time the GPU process dies.

Folivox already does nearly all of it: readers, fragmenting, a resumable journal of Takes, markup with a
Narrator plus fifteen Speakers, chapter assembly and a chaptered M4B. What it does not have is the model I
want to read with — Ming-3.0-tts, which is what I rate for English and for mixed-language books.

I cannot simply drop Ming into Folivox: Folivox pins `transformers==4.51.3` and Ming pins `4.46.3` and ships
its own model code, and Folivox is not mine to change. My earlier attempt (a Kaggle notebook) traded the
problem for worse ones: no resume, no cast, one long file, everything re-run for one changed setting.

## Solution

Two installs and one flag. `folivox-ming` adds the engine `ming` to Folivox through its provider entry point,
so `folivox book.epub --tts ming --voice en-kalynda -o out/` produces the chapter files and the chaptered M4B
exactly as Folivox would with any other local engine. The model runs in its own environment as a worker
process, so its dependencies and its GPU memory can never touch Folivox's.

Casting is Folivox's own markup over its own bundled LibriVox voices; reading speed is applied to the
assembled audio rather than to every Take, so changing pace costs seconds, not a re-render. A killed run
resumes from the Takes already on disk.

## User Stories

1. As the operator, I want to install Folivox and the Ming engine with two `pip install git+…` commands, so that there is no vendored fork to rebase.
2. As the operator, I want one command that fetches the checkpoint and builds the model's environment, so that a fresh GPU host is ready without a notebook.
3. As the operator, I want a status command that tells me which of environment, code and checkpoint is missing, so that a failed install is diagnosable without reading logs.
4. As the operator, I want `folivox providers` to list `ming`, so that I know the plugin loaded before spending GPU time.
5. As the operator, I want `--tts ming` on an ordinary Folivox render, so that no other command has to change.
6. As a listener, I want the book read by one consistent Narrator voice, so that it sounds like one person read it.
7. As a listener, I want to choose that voice from the bundled public-domain readers, so that I do not have to record anything.
8. As a listener, I want to supply my own recording and transcript as a voice, so that a book can be read in a voice I own.
9. As a listener, I want dialogue read by different Speakers from the markup, so that a novel is not one flat monologue.
10. As the operator, I want the Cast frozen in the markup before synthesis, so that a resumed run cannot silently change who speaks.
11. As the operator, I want a narrator-only mode when a book has no dialogue markup, so that the tool works on essays and non-fiction.
12. As the operator, I want EPUB, PDF and TXT accepted, so that my library is usable as it is.
13. As the operator, I want a scanned PDF refused with a clear message instead of an empty render, so that I know to OCR it myself.
14. As the operator, I want one audio file per Chapter plus one chaptered M4B, so that I can listen straight through or per chapter.
15. As a listener, I want chapter titles and durations in the M4B, so that my player can navigate the book.
16. As a listener, I want a slower or faster read of the same book, so that long books suit my commute.
17. As the operator, I want that speed change to cost no synthesis, so that the GPU is not spent twice on the same text.
18. As the operator, I want a crash or OOM-killed model process to leave the finished Takes intact, so that re-running continues rather than restarts.
19. As the operator, I want the render to tell me what died and why, so that I can tell an OOM from a bad reference clip.
20. As the operator, I want a stuck generation to time out with a resumable error, so that one hang cannot wedge an overnight run.
21. As the operator, I want a fragment in another language read in that language, so that a book with foreign phrases does not get mangled.
22. As the operator, I want the engine to choose the model's task per fragment, so that I do not have to mark languages by hand.
23. As the operator, I want the whole book at one sample rate, so that chapters concatenate without pitch drift.
24. As the operator, I want the model's real sample rate discovered at load, so that a 24 kHz answer is not written as 44.1 kHz.
25. As the operator, I want to force CPU or CUDA, so that a broken driver does not block a run.
26. As the operator, I want GPU memory released when the render stops, so that the studio and the next render are not choked.
27. As the operator, I want the engine to refuse to start with an install hint when the checkpoint is missing, so that I am not debugging a stack trace.
28. As the operator, I want Folivox's progress, warnings and chapter events unchanged, so that the studio and the CLI feel the same as with any engine.
29. As the operator, I want the studio (`folivox ui`) to run a Ming render, so that I can preview and pause without the CLI.
30. As the operator, I want to keep the per-chapter journal after a finished render, so that re-assembling with another speed is instant.
31. As the operator, I want a speed change to be repeatable and cached, so that assembling twice does not redo the work.
32. As the operator, I want speeds outside a sane range to be warned about rather than silently distorted, so that I know when to re-record instead of stretch.
33. As the operator, I want no accounts, keys, queue or billing anywhere in the path, so that the tool works offline once the checkpoint is downloaded.
34. As the operator, I want the Ming engine to be swappable for the larger `tiny` model later, so that a better GPU is the only thing between me and it.

## Implementation Decisions

**Packaging and the seam with Folivox.** A separate distribution, `folivox-ming`, with the package
`folivox_ming`, registering into Folivox's public provider registry:

```toml
[project.entry-points."folivox.providers"]
ming = "folivox_ming:register"
```

`register()` calls `ENGINES.register("ming", "folivox_ming.engine:MingEngine", …)` with a `needs` hint
naming the install command. Folivox core is never patched; only `TTSEngine`, `ENGINES`, `Voice` and `memory`
are used.

**Engine.** `MingEngine` subclasses Folivox's `TTSEngine` as a single-stage engine (`synthesize_audio`), with
`languages = ("zh", "en", "ja", "ko", "de", "es", "fr")`, `local = True`. It constructs lazily: the model is
loaded on the first synthesis, not at import, so `folivox providers` stays fast.

Folivox's local-engine path calls `start(voice)` before it reads `engine.sample_rate`, so the engine exposes
`start(voice)` and warms the model on the book's own voice there — the warm-up both pays the first-call kernel
compilation and yields the model's true output rate, which the journal then records for every Take. Folivox's
warm-up message reads `engine.build.title`, so the engine carries a `build` shim (title, id, revision).

**Worker protocol.** The model lives in a child process running the Engine environment's interpreter, speaking
one JSON object per line on stdin and one reply per line on stdout; all library output is redirected to stderr
so it cannot corrupt the channel.

| Request | Reply |
|---|---|
| `{"cmd": "load", "model_dir", "repo_dir", "device", "warmup": {"wav", "text"} \| null}` | `{"ok": true, "sample_rate", "load_s", "warm_s"}` |
| `{"cmd": "synth", "id", "text", "ref": {"wav", "text"}, "task", "language_id", "seed", "sample_rate", "out"}` | `{"ok": true, "out", "sample_rate", "audio_s", "seconds", "peak_gb"}` |
| `{"cmd": "quit"}` | — |

Failures reply `{"ok": false, "error": …}` and the process keeps serving; a worker that exits or stops
answering is reported as a worker death (with an OOM hint for exit codes −9/137) or a `SynthTimeout` telling
the operator the run resumes. The engine kills and reaps a timed-out worker rather than waiting for it.

`synth` carries the render's current `sample_rate`; if the model answers in another rate (a different task
may), the worker resamples before returning, so one book never mixes rates.

**Language and task choice.** Folivox already splits Fragments by language, so a single-stage engine sees one
language per call and decides by script: any kana → `ja`, any hangul → `ko`, otherwise any Han → `zh`, else
`en`. Mapping onto Ming:

| Language | `task` | `language_id` |
|---|---|---|
| zh | `TTS` | Chinese |
| en | `TTS` | English |
| ja / ko / de / es / fr | `MULTILINGUAL` | Japanese / Korean / German / Spanish / French |

Latin text that is not English is read as English: a Fragment carries no language hint into `synthesize_audio`,
and this is a known, accepted limitation rather than a detection heuristic.

**Synthesis call.** The worker calls `MingTTS(model_dir).synthesize(task=…, language_id=…, text=…,
prompt_wav_path=…, prompt_text=…, use_spk_emb=True, output_path=…)`, reads the produced WAV with `soundfile`,
mean-mixes to mono, and saves float32 mono for the engine. `use_spk_emb=True` keeps speaker identity steady
across Fragments. References are converted to mono 44.1 kHz WAV with ffmpeg once per voice and cached for the
render.

**Speed.** The engine ignores the `speed` argument (as the Fish engines do). Reading speed belongs to
Assembly — see ADR 0002 — so the documented render passes no `--speed`.

**Install.** `folivox_ming.install` owns what Folivox's catalogue would otherwise own:

- root `<user data>/folivox/engines`, overridable by `FOLIVOX_MING_HOME`;
- code cloned from the Ming repository at a pinned ref into `ming-3.0-tts-0.5b/repo` (its own `ming_tts`
  package is put on the worker's `sys.path`, never installed into Folivox);
- environment at `runtimes/ming-0.5b/venv` — Python 3.11 via `uv` when available, else the running 3.10/3.11,
  with Ming's `requirements.txt` plus `soundfile` (the worker's WAV reader);
- checkpoint `inclusionAI/Ming-3.0-tts-0.5b` into `ming-3.0-tts-0.5b/weights`, the `ready.json` marker written
  last and validated on both `ref` and file presence;
- `python -m folivox_ming.install --check` prints installed / missing parts and exits non-zero when incomplete.

**Assembly (to build).** A new module, `folivox_ming.assemble`, one function over Folivox's existing work
folder: reads the journal's chapters and their fully rendered Takes, applies the reading speed with
`ffmpeg -filter:a atempo`, writes speed-applied chapter FLACs into a separate speed-named folder (Folivox's own
`chapters/` stays untouched), then writes the chaptered M4B with Folivox's existing `write_m4b`, reusing its
`BookMeta` and cover from the journal. Chapter durations are measured after stretching, so markers follow the
audio. Assembly caches per (chapter signature, speed) so a second run with the same speed is a no-op. Speed
must be positive; outside roughly 0.85–1.15 the operator gets a warning that the artefacts become audible.

**CLI surface.** One addition: a command that runs Folivox's render and then Assembly, taking the book, the
work folder (default: Folivox's), the speed and the output path. Everything else stays Folivox's flags.

## Testing Decisions

Good tests assert external behaviour at one seam: a request in, observable output out. They do not reach into
private methods, do not assert on log wording, and do not need a GPU — except the one manual smoke test on
the GPU host.

**Seam 1 — the worker protocol (existing).** Spawn the worker with a stub interpreter/script that speaks the
same JSON lines and returns canned audio, then drive `MingEngine.start()` → `synthesize_audio()` → `close()`.
Covers: the load warm-up setting the engine's sample rate; a take handed back as float32 at the render's rate;
language/task chosen for en, zh, ja, ko and a French-looking Latin fragment; an error reply surfacing as
`MingError` without killing the process; a worker exiting mid-render producing a readable death message; a
worker that never answers producing `SynthTimeout` with the resume hint; `close()` leaving no process behind.

**Seam 2 — Assembly (new).** One function over a fixture work folder (a journal with two short chapters of
Takes, plus a cover), run ffmpeg for real on a second of tone. Covers: chapter files produced in order with
the chapter titles; an M4B whose chapter marks match the chapter durations; durations at speed 1.0, 1.15 and
0.85 scaling accordingly; a second run at the same speed rebuilding nothing; a changed speed not touching the
Takes; a non-positive speed refused with a clear error; an incomplete chapter skipped with a warning, never
half-assembled.

**Prior art.** Folivox's own suite is the model to follow: `tests/render_fakes.py` for stand-ins imported by
name inside a child process, `tests/test_decided_pipeline.py` for driving the pipeline with a minimal `Events`
double and asserting on the journal and the produced files, and `tests/fixtures` for small books.

**Not in the suite.** A real synthesis needs the checkpoint and a GPU: one manual smoke test on the GPU host
(one chapter, one bundled voice) confirms the install, the load timing and the true sample rate. It is
documented, not automated.

## Out of Scope

- The Kaggle notebook runner: the engine plus Folivox's resumable journal replace it. Running on a rented GPU
  by other means (Docker, a VM) is deployment, not this spec.
- Per-character derived or pitch-shifted voices (the notebook's fake-voice trick); the Cast uses real
  reference clips only.
- Expression tags: Folivox writes them for Fish engines only; Ming reads plain text.
- Cloud engines, keys, billing, cost caps — none are on this path.
- Accounts, a queue, multi-tenancy, a database: one operator, folders and a journal.
- `Ming-3.0-tts-tiny` support: the worker is written so the model id and environment can change, but this spec
  ships the 0.5B checkpoint.
- A GUI: `folivox ui` is the studio; no Gradio panel of our own.
- Serving the model through `sgl-omni`/vLLM instead of the worker process.

## Further Notes

- The model's output rate is not stated in its README; the warm-up is the only trustworthy source, which is
  why the engine learns it there instead of hard-coding it.
- 0.5B is chosen for a 16 GB card (~3 GB weights, ~1.5 s of compute per second of speech). `tiny` is a 7.9B
  MoE needing ~18 GB and a different Python/`transformers` pin — hence the environment is isolated per model.
- Reading speed is bounded by `atempo` quality, not by the model: past ~±15% the framing is audible and a
  re-render at another pace (a different prompt) is the honest answer.
- Licence: Ming is MIT; Folivox is Apache-2.0. No attribution trailer is required in the audio.

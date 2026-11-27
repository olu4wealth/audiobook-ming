# folivox-ming

Ming-3.0-tts as a speech engine for [Folivox](https://github.com/vatavale/folivox) audiobooks.

Folivox already reads EPUB/PDF/TXT, chunks the text, caches one take per fragment, keeps a resumable
journal, casts a narrator plus characters through its markup, and assembles a chaptered M4B. This package
adds only the missing piece: the engine. It is not a fork — it registers `ming` into Folivox's provider
registry (see [ADR 0001](docs/adr/0001-ming-as-out-of-tree-provider.md)).

## Install

```bash
pip install git+https://github.com/vatavale/folivox          # the host tool
pip install git+https://github.com/olu4wealth/audiobook-ming # this engine
python -m folivox_ming.install                               # ~3 GB checkpoint + private environment
folivox providers                                            # `ming` is listed now
```

The engine repository is private until a real GPU render has passed; while it is, `pip install` from it needs
a token or an authenticated `git`. Clone it and `pip install .` instead.

The model runs in its own environment (`transformers==4.46.3`, its own torch), not Folivox's: the two cannot
share one. `python -m folivox_ming.install --check` says what is in place.

Ming-3.0-tts-0.5b needs a CUDA GPU (~3 GB of VRAM, comfortable on a 16 GB card) and is time-consuming
without one. There is no CPU path worth waiting for.

## Render

```bash
folivox markup book.epub                     # cast: narrator + characters, saved as book.markup.json
python -m folivox_ming render book.epub --out out/book.m4b --voice en-kalynda --speed 1.1
```

That one command renders with Ming through Folivox (plan, journal, Takes, resume — all Folivox's) and then
assembles the result: one file per chapter under `out/book.chapters/speed-1.1/`, plus the chaptered
`out/book.m4b`. Chapter files are cached per speed, so re-running with a different `--speed` reuses every
Take and costs seconds.

Everything else is Folivox's: `--chapters`, `--preview`, resuming a stopped render, `folivox ui` for the
studio. Voices are the bundled LibriVox readers or your own recording plus transcript — the engine clones
whatever reference the markup names, one voice per speaker.

Reading speed is **not** passed to Folivox (`--speed` only re-keys takes, and this model has no speed
parameter). It is applied at assembly with `atempo`, so changing it never re-synthesises anything — see
[ADR 0002](docs/adr/0002-speed-applied-at-assembly.md). `--speed` outside 0.85–1.15 warns, because the
stretch starts to be audible. A stopped or crashed render resumes: run the same command again.

## Layout

| Path | What it is |
|---|---|
| `src/folivox_ming/engine.py` | `MingEngine` (Folivox `TTSEngine`) + worker client |
| `src/folivox_ming/worker.py` | the model process: JSON lines in, JSON lines out |
| `src/folivox_ming/install.py` | environment, checkpoint, `--check` |
| `src/folivox_ming/assemble.py` | Assembly: stretch chapters to a reading speed, write the chaptered M4B |
| `src/folivox_ming/cli.py` | `python -m folivox_ming render`: render, then assemble |
| `GLOSSARY.md` | the words this project and Folivox share |
| `docs/adr/` | the decisions behind the shape of it |

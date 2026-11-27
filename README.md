# folivox-ming

Ming-3.0-tts as a speech engine for [Folivox](https://github.com/vatavale/folivox) audiobooks.

Folivox already reads EPUB/PDF/TXT, chunks the text, caches one take per fragment, keeps a resumable
journal, casts a narrator plus characters through its markup, and assembles a chaptered M4B. This package
adds only the missing piece: the engine. It is not a fork — it registers `ming` into Folivox's provider
registry (see [ADR 0001](docs/adr/0001-ming-as-out-of-tree-provider.md)).

## Install

```bash
pip install git+https://github.com/vatavale/folivox          # the host tool
pip install git+https://github.com/<you>/audiobook-ming      # this engine
python -m folivox_ming.install                               # ~3 GB checkpoint + private environment
folivox providers                                            # `ming` is listed now
```

The model runs in its own environment (`transformers==4.46.3`, its own torch), not Folivox's: the two cannot
share one. `python -m folivox_ming.install --check` says what is in place.

Ming-3.0-tts-0.5b needs a CUDA GPU (~3 GB of VRAM, comfortable on a 16 GB card) and is time-consuming
without one. There is no CPU path worth waiting for.

## Render

```bash
folivox markup book.epub                     # cast: narrator + characters, saved as book.markup.json
folivox book.epub --tts ming --voice en-kalynda -o out/
```

Everything else is Folivox's: `--chapters`, `--preview`, resuming a stopped render, `folivox ui` for the
studio. Voices are the bundled LibriVox readers or your own recording plus transcript — the engine clones
whatever reference the markup names, one voice per speaker.

Reading speed is **not** passed to Folivox (`--speed` only re-keys takes, and this model has no speed
parameter). It belongs to assembly, where `atempo` applies it to the finished audio — see
[ADR 0002](docs/adr/0002-speed-applied-at-assembly.md). The assembly stage is the next piece of work; until
it lands, Folivox's own chapter/M4B output is used as-is, at the model's natural pace.

## Layout

| Path | What it is |
|---|---|
| `src/folivox_ming/engine.py` | `MingEngine` (Folivox `TTSEngine`) + worker client |
| `src/folivox_ming/worker.py` | the model process: JSON lines in, JSON lines out |
| `src/folivox_ming/install.py` | environment, checkpoint, `--check` |
| `GLOSSARY.md` | the words this project and Folivox share |
| `docs/adr/` | the decisions behind the shape of it |

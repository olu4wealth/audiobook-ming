# ADR 0001 — Ming is an out-of-tree Folivox provider

- Status: accepted
- Date: 2026-11-27

## Context

Folivox already does what this project needs: EPUB/PDF/TXT readers, sentence-level chunking, per-fragment
takes with a resumable journal, markup with a narrator plus 15 speakers, chapter assembly and a chaptered
M4B, a studio, and a provider registry. Rewriting it would be a second product with none of its testing.

It has no Ming engine, and it is not ours to change (we do not own the upstream repository).

Two hard incompatibilities make "just add a class to the Folivox tree" impractical anyway:

- Folivox pins `transformers==4.51.3`; Ming-3.0-tts pins `transformers==4.46.3` and ships its own model
  code (`ming_tts`), not a Hub `transformers` architecture.
- Folivox's engine catalogue, weight verification and `folivox engines install` are keyed on *its* catalog
  entries; Ming would need a new entry, a pinned revision and checksums upstream, per release.

## Decision

Ship Ming as a separate installable package, `folivox-ming`, that registers itself into Folivox through the
public `folivox.providers` entry-point group:

```toml
[project.entry-points."folivox.providers"]
ming = "folivox_ming:register"      # → ENGINES.register("ming", "folivox_ming.engine:MingEngine", …)
```

The engine lives in its **own** environment (the Engine environment) and is driven as a Worker over JSON
lines, the same way Folivox drives its Fish engines: the model is imported by a child process, never by the
Folivox interpreter, so its torch and `transformers` versions cannot collide with Folivox's.

Folivox core is used exactly as published. Nothing imports `folivox.*` internals beyond the documented
`TTSEngine`, `ENGINES`, `Voice` and `memory` surfaces.

## Consequences

- `pip install git+https://github.com/<user>/audiobook-ming` next to Folivox is the whole integration; a
  Folivox upgrade is a reinstall, not a merge.
- Ming can pin whatever torch/`transformers` it wants, and can be swapped for `tiny` later without touching
  Folivox.
- The provider owns what Folivox's catalogue would have given it: installing the weights (Hugging Face),
  creating the environment, the `is_ready` check and the "install it first" error message.
- A crash or an OOM kill inside Ming is a Worker death with a readable message, not a dead audiobook render.
- Cost: `folivox engines …` does not list Ming; the Ming install command carries that job
  (`python -m folivox_ming.install`).

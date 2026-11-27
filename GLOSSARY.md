# Audiobook (Ming)

Turns a **Book** into a chaptered audiobook with Folivox, using the **Ming provider** as its speech engine.
Personal tool: one operator, one GPU host, no accounts and no queue — but nothing here assumes a single
process, so it can become a service later.

The words below are Folivox's, except where marked *(ours)*.

## Language

**Book**:
The input document a render reads — EPUB, PDF or TXT. Folivox converts it to its own text form first.
_Avoid_: source, document, input, file

**Chapter**:
A top-level division of a Book; becomes one chapter file and one chapter marker in the finished audiobook.
_Avoid_: section, part, track

**Fragment**:
The smallest unit of text Folivox synthesizes — a sentence-sized run with one speaker and one language.
Cached individually, so a resumed render re-reads only fragments it does not have.
_Avoid_: chunk, line, segment

**Take**:
One stored synthesis of one Fragment, at `audio/<key>.flac`, keyed on the settings, the voice id and the text.
_Avoid_: clip, sample, render

**Voice**:
A reference recording plus its transcript; the engine clones its timbre. Folivox bundles public-domain
LibriVox readers (`en-kalynda`, `en-ben-tucker`, …) and accepts your own with `folivox voices add`.
_Avoid_: speaker model, preset, ref audio

**Narrator**:
Speaker id 0 in the markup: reads every Fragment not given to a character.
_Avoid_: default voice, main voice

**Speaker**:
A numbered voice in the Book's markup, ids 1–15; each may name a Voice or fall back to the Narrator's.
_Avoid_: character, role, actor

**Markup**:
`<book>.markup.json`, written by `folivox markup`: the speakers, their spans and the expression tags, with a
signature of each chapter's text. It is the frozen Cast — fixed before any audio exists, never inferred
during a render.
_Avoid_: cast file, script, meta, annotations

**Cast**:
The speakers and Voices the markup assigns for one audiobook, one per speaking character.
_Avoid_: characters, voices, roles

**Cast mode**:
Which Voices a render uses — narrator only, or narrator plus the characters the markup defines.
_Avoid_: multi-voice, single voice

**Ming provider** *(ours)*:
The out-of-tree Folivox engine `ming`, registering `folivox_ming.engine:MingEngine` through the
`folivox.providers` entry point. Reads text with Ming-3.0-tts-0.5b; never patches Folivox.
_Avoid_: plugin, fork, integration, model

**Worker** *(ours)*:
The child process that holds the Ming model, running the interpreter of the Engine environment and talking
JSON lines over pipes. A crash or an out-of-memory kill takes down only the Worker.
_Avoid_: server, backend, subprocess

**Engine environment** *(ours)*:
The private virtual environment at `engines/runtimes/ming-0.5b/venv`, with the Python, torch and
`transformers==4.46.3` Ming needs — versions Folivox's own environment must not carry.
_Avoid_: venv, conda env, dependencies

**Render**:
A Folivox run over a Book: plans the Fragments, synthesizes the missing Takes, then assembles the audio.
Resumable — a stopped render continues from the Takes already on disk.
_Avoid_: job, run, pass

**Assembly**:
The stage that concatenates a chapter's Takes into the chapter files and the finished chaptered audiobook.
_Avoid_: stitching, merge, export

**Reading speed** *(ours)*:
A multiplier on playback pace, applied to the audio at Assembly (`atempo`), never to the Takes and never part
of their key. See [ADR 0002](docs/adr/0002-speed-applied-at-assembly.md).
_Avoid_: tempo, rate, tts speed, --speed

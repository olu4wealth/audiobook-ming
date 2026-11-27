# ADR 0002 — Reading speed is applied at assembly, not at synthesis

- Status: accepted
- Date: 2026-11-27

## Context

Ming-3.0-tts has no speed parameter on the cloning path: `synthesize(task="TTS", prompt_wav_path=…,
prompt_text=…)` takes text, a reference and controls. Only its `BASIC` task exposes a coarse speed control
(`"慢速"` / `"中"` / `"快速"`), and that task accepts no reference text — it is not the voice-cloning path we
use.

Folivox, meanwhile, puts the reading speed into the take key: for a non-CosyVoice engine
(`pipeline.py`, `settings = f"{opts.tts}|speed={opts.speed}"`). A speed baked into the key means any change of
speed throws away every take and re-renders the whole book — hours of GPU time for a 10% pace change.

## Decision

The Takes are always synthesized at the model's natural pace, with Folivox rendering at speed `1.0`. Reading
speed is a property of **Assembly**: the stage that builds the chapter files and the M4B applies
`ffmpeg -filter:a atempo=<speed>` to the assembled audio.

- The engine ignores the `speed` argument Folivox passes (as the Fish engines already do), so a speed change
  never reaches the model.
- Speed is therefore *not* part of any take key: takes stay valid across speed changes.
- The documented command is a render without `--speed`; speed is given to the assembly step.

## Consequences

- Changing pace is cheap: seconds of `atempo`, no re-synthesis, no GPU.
- Pace is uniform per chapter and per book; per-character or per-span pace is out of scope.
- `atempo` is a time-stretch, not a different performance: phrasing, breaths and pauses all shift with it.
  That is acceptable for 0.85–1.15; outside that range the artefacts become audible and a re-render at a
  different pace (a different model prompt, not a different filter) is the honest option.
- Assembly owns speed, so the finished files and the chapter files always agree; nothing reads speed back
  from the journal.
- Implemented as `folivox_ming.assemble` plus the `render` command: stretched chapter files land in
  `<book>.chapters/speed-<n>/` (one folder per speed, so two speeds coexist), the M4B's chapter marks are
  measured from the stretched audio, and each stretched chapter is cached in the journal against the
  chapter's signature and the speed — a second run at the same speed rebuilds nothing.

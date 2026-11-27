"""One command: render a Book with the Ming engine, then assemble it at the chosen reading speed.

Everything the render does is Folivox's own — plan, journal, Takes, resume — so this module only chooses the
engine, resolves the Narrator voice, and hands the work folder to Assembly afterwards (ADR 0002).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from folivox.audio.assemble import BookMeta
from folivox.pipeline import Events, RenderOptions, render
from folivox.readers import read_book

from .assemble import assemble, check_speed

ENGINE = "ming"


class ConsoleEvents(Events):
    """Folivox's progress, minus the styling: enough for a terminal run and for a log file."""

    def stage(self, stage: str, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def warning(self, message: str) -> None:
        print(f"warning: {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="folivox_ming",
        description="Render an audiobook with the Ming engine and write the chaptered M4B. "
                    "Reading speed is applied at Assembly, so changing it never re-synthesises a Take.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    cmd = subcommands.add_parser("render", help="render a Book with Ming, then assemble it")
    cmd.add_argument("book", type=Path, help="the Book to read (EPUB, digitally created PDF, or TXT)")
    cmd.add_argument("--out", type=Path, required=True, help="the chaptered M4B to write")
    cmd.add_argument("--work-dir", type=Path, default=None,
                     help="Folivox work folder holding the journal and Takes (default: beside --out)")
    cmd.add_argument("--speed", type=float, default=1.0,
                     help="reading speed, applied at Assembly (default 1.0)")
    cmd.add_argument("--device", default="auto", help="auto, cuda or cpu")
    cmd.add_argument("--voice", default="russlib", help="Narrator voice id from the Folivox catalog")
    cmd.add_argument("--bitrate", default="64k", help="M4B audio bitrate (default 64k)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = ConsoleEvents()
    try:
        check_speed(args.speed, events)
        path = render_and_assemble(args, events)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def render_and_assemble(args: argparse.Namespace, events: Events) -> Path:
    book = read_book(args.book)
    for note in getattr(book, "warnings", []):
        events.warning(note)
    narrator = _voice(args.voice)
    opts = RenderOptions(output=args.out, work_dir=args.work_dir, tts=ENGINE, device=args.device,
                         voice_wav=Path(narrator.path), bitrate=args.bitrate)
    render(book, opts, events)
    meta = BookMeta(title=book.title or args.book.stem, author=book.author, year=book.year,
                    comment=book.annotation, narrator=narrator.name)
    return assemble(opts.resolved_work_dir(), args.out, speed=args.speed, meta=meta, cover=book.cover,
                    events=events, bitrate=args.bitrate)


def _voice(voice_id: str):
    from folivox import voices

    entry = voices.find(voice_id)
    if entry is None:
        raise ValueError(f"unknown Narrator voice {voice_id!r}: run `folivox voices` to list them")
    return entry

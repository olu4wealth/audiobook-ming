"""Ming-3.0-tts as an out-of-tree Folivox engine.

Folivox discovers this package through the ``folivox.providers`` entry point, which calls :func:`register`
the first time it resolves providers. Nothing here imports Folivox at module level: the plugin must stay
importable when the host is missing, so the registry can explain itself instead of crashing the CLI.
"""

from __future__ import annotations

__version__ = "0.1.0"

ENGINE_NAME = "ming"
TARGET = "folivox_ming.engine:MingEngine"


def register() -> None:
    """Add ``--tts ming`` to Folivox."""
    from folivox.tts import ENGINES

    ENGINES.register(
        ENGINE_NAME,
        TARGET,
        "Ming-3.0-tts-0.5b (MIT, local GPU): zero-shot cloning, zh en de es fr ko ja",
        needs="python -m folivox_ming.install",
    )

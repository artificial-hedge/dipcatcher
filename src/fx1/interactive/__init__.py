"""fxi — the interactive front door for fx-1.

fx-1 recognizes the planned model identifiers ``fx1`` and ``fx1-lite``.
Each can use a configured endpoint (API key + base URL) managed by
``fx1.interactive.profiles`` and injected into the dipcatcher harness.
Neither identifier implies released weights or a hosted deployment. See
``fx1.interactive.app`` for the CLI entry point and
``fx1.interactive.wizard`` for first-run onboarding.

The orchestration surface lives alongside the classic shell:

- ``concierge`` — chat, /superpower, background web research, /flash.
- ``superpower`` — capability registry, planner, coordinator.
- ``approvals`` — fail-closed gates for consequential actions and
  private-data sharing.

Submodules are imported on access so ``dipcatcher chat --help`` and
``from fx1.interactive.profiles import MODELS`` do not load the serve stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "actions",
    "app",
    "approvals",
    "concierge",
    "main",
    "orb",
    "profiles",
    "shell",
    "superpower",
    "wizard",
]


def __getattr__(name: str) -> Any:
    if name in {"app", "main"}:
        from fx1.interactive.app import app, main

        return app if name == "app" else main
    if name in __all__:
        return __import__(f"fx1.interactive.{name}", fromlist=[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

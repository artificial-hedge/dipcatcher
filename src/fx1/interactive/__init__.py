"""fxi — the interactive front door for fx-1.

fx-1 ships two hosted model names (``fx1`` and ``fx1-lite``); each has one
endpoint (API key + base URL) managed by ``fx1.interactive.profiles`` and
injected into the dipcatcher harness. See ``fx1.interactive.app`` for the
CLI entry point and ``fx1.interactive.wizard`` for first-run onboarding.

The orchestration surface lives alongside the classic shell:

- ``concierge`` — chat, /superpower, background web research, /flash.
- ``superpower`` — capability registry, planner, coordinator.
- ``approvals`` — fail-closed gates for consequential actions and
  private-data sharing.
"""

# The orchestration modules (``concierge``, ``superpower``, ``approvals``)
# ship in a later commit; they are intentionally absent from this import so
# the package stays importable while the surface is staged. Callers reach
# them lazily via importlib (see fx1.interactive.app / wizard).
from fx1.interactive import actions, orb, profiles, shell, wizard
from fx1.interactive.app import app, main

__all__ = [
    "actions",
    "app",
    "main",
    "orb",
    "profiles",
    "shell",
    "wizard",
]

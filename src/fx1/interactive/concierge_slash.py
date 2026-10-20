"""Slash-command helpers for the concierge."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any

# Console loop: faster polls while a background job is active so progress and
# cancellation stay snappy; idle polls stay cheap for the CPU.
ACTIVE_JOB_POLL_S = 0.04
MAX_PROGRESS_EVENTS_PER_TICK = 48

BUDGET_DIMENSIONS: dict[str, str] = {
    "wall": "max_wall_s",
    "harness": "max_harness_steps",
    "calls": "max_model_calls",
    "retries": "max_retries_per_call",
    "sources": "max_web_sources",
    "tokens": "max_prompt_tokens",
}

# Codex-style aliases: the canonical names stay in HELP_TEXT.
SLASH_ALIASES: dict[str, str] = {
    "quit": "exit",
    "?": "help",
}


def slash_commands() -> tuple[str, ...]:
    """Registered ``/`` commands derived from ``_cmd_*`` handlers."""
    from fx1.interactive.concierge import Concierge

    return tuple(
        sorted(
            name[len("_cmd_") :].replace("_", "-")
            for name in dir(Concierge)
            if name.startswith("_cmd_")
        )
    )


def suggest_slash(command: str) -> str:
    matches = get_close_matches(command, slash_commands(), n=3, cutoff=0.55)
    if not matches:
        return "try /help"
    listed = ", ".join(f"/{name}" for name in matches)
    return f"did you mean {listed}? try /help"


def doctor_next_actions(status: dict[str, Any], model: str) -> list[str]:
    """Actionable follow-ups for ``/doctor`` — presence-only, no secrets."""
    actions: list[str] = []
    active_unset = True
    models = status.get("models")
    if isinstance(models, list):
        for row in models:
            if isinstance(row, dict) and row.get("model") == model:
                active_unset = row.get("status") == "unset"
                break
    if active_unset:
        actions.append(f"/keys set {model}  (store API key + endpoint URL; input is masked)")
        actions.append(f"/keys test {model}  (identity probe once an endpoint is stored)")
    else:
        actions.append(f"/keys test {model}  (confirm requested vs served model identity)")
        actions.append("/superpower <goal> [--sync]  (plan + run eligible capabilities)")
    if status.get("family_catalog") == "not loaded yet":
        actions.append("/families  (load the research-family catalog the planner consults)")
    actions.append("/help")
    return actions

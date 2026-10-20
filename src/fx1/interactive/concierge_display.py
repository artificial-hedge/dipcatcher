"""Display, scrubbing, and approval-tap helpers for the concierge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.model_ux import public_error
from fx1.interactive.superpower import SuperpowerResult
from fx1.interactive.superpower_glue import evidence_status_line


def safe_slash_error(exc: Exception) -> str:
    """Operator-facing slash errors. Do not echo credential-shaped text."""
    return public_error(exc)


def stdout_color() -> bool:
    from fx1.interactive.terminal import detect_caps

    try:
        return bool(detect_caps().color)
    except (OSError, ValueError):
        return False


def int_field(obj: Any, name: str, default: int) -> int:
    limit = getattr(obj, name, default)
    if isinstance(limit, bool) or not isinstance(limit, int):
        return default
    return limit


def superpower_memory(result: SuperpowerResult) -> str:
    parts = (
        scrub_user(result.summary or ""),
        evidence_status_line(result),
        scrub_user(result.explain()),
    )
    text = "\n\n".join(part for part in parts if part.strip())
    return memory_text(text)


def memory_text(text: str) -> str:
    if not text.strip():
        return ""
    try:
        validate_fx1_output(text)
    except Fx1HonestyError:
        return "answer withheld by the fx-1 honesty gate"
    return text


def scrub_user(text: str) -> str:
    from fx1.interactive.model_ux import scrub_transcript

    return scrub_transcript(text)


def scrub_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return scrub_user(value)


def scrub_patch_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_user(value)
    if isinstance(value, list):
        return [scrub_patch_value(item) for item in value]
    return value


def scrub_patch(patch: dict[str, Any]) -> dict[str, Any]:
    return {key: scrub_patch_value(value) for key, value in patch.items()}


def for_display(text: str, *, keep_color: bool | None = None) -> str:
    """Sanitize for the terminal after callers have validated the text."""
    from fx1.interactive.redact import redact_text
    from fx1.interactive.terminal import sanitize_for_terminal

    color = stdout_color() if keep_color is None else keep_color
    return sanitize_for_terminal(redact_text(text), keep_sgr=color)


class ApprovalTap:
    """Forwards approvals and emits the key plus the decision, never the question."""

    def __init__(self, gate: ApprovalGate, emit: Callable[[str, Mapping[str, Any]], None]) -> None:
        self._gate = gate
        self._emit = emit

    def ask(
        self,
        question: str,
        *,
        kind: str = "action",
        key: str | None = None,
        bounds: Any = None,
        timeout_s: float | None = None,
    ) -> bool:
        approved = bool(
            self._gate.ask(question, kind=kind, key=key, bounds=bounds, timeout_s=timeout_s)
        )
        self._emit(
            "concierge.approval",
            {"key": key or "", "kind": kind, "approved": approved},
        )
        return approved

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gate, name)

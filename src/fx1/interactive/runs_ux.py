"""Read-only ``/runs`` rendering over a :class:`JobLedger`.

Listing and detail use ``list`` and ``get`` only. Nothing here transitions a
job or rewrites a record.
"""

from __future__ import annotations

from typing import Any

from fx1.interactive.atomicstore import StoreCorruptError
from fx1.interactive.jobs import JobNotFound
from fx1.interactive.redact import redact_text
from fx1.interactive.terminal import sanitize_for_terminal

_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "interrupted"})
_KNOWN = _TERMINAL | frozenset({"queued", "running", "cancelling"})
_LIST_LIMIT = 20
_FOCUS_LIMIT = 200


def render_runs(ledger: Any, focus: str = "", *, limit: int = _LIST_LIMIT) -> str:
    """Newest jobs, or one job's steps when ``focus`` is an id prefix."""
    token = focus.strip()
    if token:
        return _focused(ledger, token)
    records = _listed(ledger, limit)
    if not records:
        return "no jobs"
    return "\n".join(_job_line(record) for record in records)


def _focused(ledger: Any, token: str) -> str:
    matches = [record for record in _listed(ledger, _FOCUS_LIMIT) if _matches(record, token)]
    if not matches:
        return _by_id(ledger, token)
    exact = [record for record in matches if _job_id(record) == token]
    if len(matches) == 1 or len(exact) == 1:
        chosen = exact[0] if exact else matches[0]
        return _detail(chosen)
    return "\n".join(_job_line(record) for record in matches)


def _by_id(ledger: Any, token: str) -> str:
    try:
        return _detail(ledger.get(token))
    except (JobNotFound, StoreCorruptError, OSError, ValueError):
        return f"no job {_safe(token)}"


def _listed(ledger: Any, limit: int) -> list[Any]:
    records = ledger.list(limit=limit)
    if not isinstance(records, list):
        return []
    return records


def _matches(record: Any, token: str) -> bool:
    job_id = _job_id(record)
    return job_id == token or job_id.startswith(token)


def _detail(record: Any) -> str:
    lines = [_job_line(record), f"goal: {_safe(getattr(record, 'goal', ''))}"]
    steps = _steps(record)
    if not steps:
        lines.append("no steps")
        return "\n".join(lines)
    for index, step in enumerate(steps, start=1):
        lines.append(_step_line(index, step))
        lines.extend(_receipt_line(path) for path in _receipts(step))
    return "\n".join(lines)


def _job_line(record: Any) -> str:
    ok, total = _counts(record)
    return (
        f"{_job_id(record)} {_safe(getattr(record, 'kind', ''))} {_state_name(record)} "
        f"steps {ok}/{total} started {_safe(getattr(record, 'created_at', ''))} "
        f"finished {_finished(record)}"
    )


def _step_line(index: int, step: Any) -> str:
    code = getattr(step, "exit_code", None)
    shown = "-" if code is None else _safe(code)
    name = _safe(getattr(step, "name", ""))
    return f"step {index} {name} {_state_name(step)} exit {shown}"


def _receipt_line(path: str) -> str:
    return f"receipt {_safe(path)}"


def _counts(record: Any) -> tuple[int, int]:
    steps = _steps(record)
    ok = sum(1 for step in steps if _state_name(step) == "succeeded")
    return ok, len(steps)


def _finished(record: Any) -> str:
    if _state_name(record) not in _TERMINAL:
        return "-"
    return _safe(getattr(record, "updated_at", ""))


def _steps(record: Any) -> list[Any]:
    steps = getattr(record, "steps", None)
    if not isinstance(steps, list):
        return []
    return steps


def _receipts(step: Any) -> list[str]:
    paths = getattr(step, "receipt_paths", None)
    if not isinstance(paths, list):
        return []
    return [path for path in paths if isinstance(path, str)]


def _job_id(record: Any) -> str:
    value = getattr(record, "job_id", "")
    if isinstance(value, str):
        return value
    return ""


def _state_name(record: Any) -> str:
    state = getattr(record, "state", "")
    value = getattr(state, "value", state)
    text = str(value)
    if text in _KNOWN:
        return text
    return "unknown"


def _safe(value: object) -> str:
    text = sanitize_for_terminal(redact_text(str(value)))
    collapsed = " ".join(text.split())
    return collapsed or "-"

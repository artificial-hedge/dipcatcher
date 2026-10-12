"""Session list, export, resume text, and per-turn metadata.

Writes go through the session store and :mod:`fx1.interactive.atomicstore`.
A revision conflict is reported and is never forced over a newer file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fx1.interactive.atomicstore import atomic_write_text
from fx1.interactive.model_ux import public_error, scrub_transcript
from fx1.interactive.sessions import SessionConflict, SessionError

_OUTCOMES = {
    "ok": "ok",
    "withheld": "withheld-by-honesty-gate",
    "error": "error",
    "cancelled": "cancelled",
    "budget": "budget",
    "no_endpoint": "error",
    "none": "ok",
}


def turn_outcome(status: str) -> str:
    """Map a concierge status onto a stored turn outcome."""
    return _OUTCOMES.get(status, "error")


def turn_detail(label: str) -> str:
    """Short redacted label. The operator's text is not included."""
    cleaned = scrub_transcript(label).replace("\n", " ").strip()
    return cleaned[:120] or "turn"


def record_turn(
    store: Any,
    record: dict[str, Any],
    *,
    model: str,
    served_model: str | None,
    usage: Mapping[str, Any] | None,
    outcome: str,
    detail: str,
) -> str | None:
    """Append turn metadata. Returns a warning instead of raising."""
    try:
        store.append_turn(
            record,
            model=model,
            served_model=served_model,
            usage=usage,
            outcome=turn_outcome(outcome),
            detail=turn_detail(detail),
        )
    except SessionConflict:
        return "session changed on disk; this turn was not saved over a newer revision"
    except SessionError as exc:
        return f"session was not saved: {public_error(exc)}"
    return None


def stage_record(
    record: dict[str, Any],
    *,
    model: str,
    messages: list[dict[str, Any]],
    job_running: bool,
    job_error: str,
    job_present: bool,
    budget: Mapping[str, Any],
    approvals: list[dict[str, Any]],
    attachments: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Update the in-memory session before :meth:`SessionStore.save`."""
    record["model"] = model
    record["messages"] = list(messages)
    record["job"] = _job_blob(record.get("job"), job_running, job_error, job_present)
    if budget:
        record["budget"] = dict(budget)
    record["approvals"] = list(approvals)
    if attachments is not None:
        record["attachments"] = [dict(item) for item in attachments]


def format_session_line(record: Mapping[str, Any], model: str) -> str:
    """One line: id, model, job state, and turn count."""
    job = record.get("job") if isinstance(record.get("job"), dict) else {}
    state = job.get("state", "idle") if isinstance(job, dict) else "idle"
    turns = record.get("turns")
    count = len(turns) if isinstance(turns, list) else 0
    return f"session {record.get('id')} model {model} job {state} turns {count}"


def format_session_list(rows: Sequence[Mapping[str, Any]]) -> str:
    """Newest-first rows: id prefix, model, status, turns, updated."""
    if not rows:
        return "no saved sessions"
    return "\n".join(_list_row(row) for row in rows)


def interrupted_hint() -> str:
    """Shown after resume. The job is not retried."""
    return "inspect it with /evidence and /sessions; the job was not restarted"


def export_session(
    store: Any,
    record: Mapping[str, Any],
    kind: str,
    dest: Path | None,
    config_dir: Path,
) -> tuple[Path | None, str]:
    """Write a redacted export. Refuses to replace an existing path."""
    target = _export_target(record, kind, dest, config_dir)
    if target.exists():
        return None, f"refusing to overwrite {target}"
    body = _export_body(store, record, kind)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, body)
    return target, f"exported session to {target}"


def _job_blob(current: Any, running: bool, error: str, present: bool) -> dict[str, Any]:
    state = "running" if running else "idle"
    if present and not running:
        state = "finished" if not error else "idle"
    blob = dict(current) if isinstance(current, dict) else {}
    if blob.get("state") == "interrupted" and state == "idle":
        state = "interrupted"
    blob["state"] = state
    blob.setdefault("resume", "manual" if state == "interrupted" else "none")
    blob.setdefault("steps_completed", [])
    return blob


def _list_row(row: Mapping[str, Any]) -> str:
    ident = str(row.get("id") or "")
    return (
        f"{ident[:8]}  {row.get('model')}  {row.get('status')}  "
        f"turns {row.get('n_turns', 0)}  {row.get('updated_at')}"
    )


def _export_target(
    record: Mapping[str, Any], kind: str, dest: Path | None, config_dir: Path
) -> Path:
    if dest is not None:
        return dest
    ext = "json" if kind == "json" else "md"
    return config_dir / "exports" / f"{record.get('id')}.{ext}"


def _export_body(store: Any, record: Mapping[str, Any], kind: str) -> str:
    if kind == "json":
        return str(store.export_json(record))
    return str(store.export_markdown(record))

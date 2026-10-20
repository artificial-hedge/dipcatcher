"""Final assertions against the journey workspace. No store is rewritten."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.fx1.journeys_pty.support import CORRUPT_ID
from tests.fx1.journeys_pty.textutil import snippet, substitute


def final_problem(
    final: dict[str, Any],
    *,
    paths: dict[str, Path],
    transcript: str,
    exit_code: int | None,
    ended: bool,
    stub_url: str,
) -> str | None:
    """Return a one-line failure, or None when every declared check holds."""
    work = str(paths["work"])
    resolved = _resolve(final, stub_url=stub_url, work=work)
    problem = _process_problem(resolved, exit_code=exit_code, ended=ended)
    if problem:
        return problem
    problem = _session_problem(resolved, paths["sessions"])
    if problem:
        return problem
    return _text_problem(resolved, paths, transcript)


def session_records(directory: Path) -> list[dict[str, Any]]:
    """Valid session objects. Corrupt and non-JSON files are skipped."""
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if ".corrupt-" in path.name:
            continue
        row = _read_object(path)
        if row is not None:
            rows.append(row)
    return rows


def newest_session(directory: Path) -> dict[str, Any] | None:
    rows = session_records(directory)
    if not rows:
        return None
    return max(rows, key=lambda row: str(row.get("updated_at", "")))


def job_state_of(record: dict[str, Any] | None) -> str:
    if record is None:
        return ""
    job = record.get("job")
    if isinstance(job, dict):
        return str(job.get("state", ""))
    return ""


def _resolve(final: dict[str, Any], *, stub_url: str, work: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in final.items():
        if isinstance(value, list):
            resolved[key] = [substitute(item, stub_url=stub_url, work=work) for item in value]
        else:
            resolved[key] = value
    return resolved


def _process_problem(final: dict[str, Any], *, exit_code: int | None, ended: bool) -> str | None:
    if "exit_code" in final and exit_code != final["exit_code"]:
        return f"process exit {exit_code} != {final['exit_code']}"
    if "ended" in final and bool(ended) != bool(final["ended"]):
        return f"session ended={ended}, expected {final['ended']}"
    return None


def _session_problem(final: dict[str, Any], sessions: Path) -> str | None:
    record = newest_session(sessions)
    if final.get("session_exists") and record is None:
        return "no session file was written"
    if "job_state" in final and job_state_of(record) != final["job_state"]:
        return f"job state {job_state_of(record)!r} != {final['job_state']!r}"
    if final.get("corrupt_quarantined"):
        problem = _quarantine_problem(sessions)
        if problem:
            return problem
    blob = _read_tree(sessions)
    return _list_problem(blob, final, "session_contains", "session_forbids", "session")


def _quarantine_problem(sessions: Path) -> str | None:
    original = sessions / f"{CORRUPT_ID}.json"
    moved = [path.name for path in sessions.iterdir() if ".corrupt-" in path.name]
    if original.exists() or not moved:
        return "corrupt session file was reported but not quarantined"
    return None


def _text_problem(final: dict[str, Any], paths: dict[str, Path], transcript: str) -> str | None:
    exports = _read_tree(paths["config"] / "exports")
    history = _read_file(paths["config"] / "history")
    logs = _read_tree(paths["config"] / "logs")
    checks = (
        _list_problem(exports, final, "export_contains", "export_forbids", "export"),
        _list_problem(history, final, "", "history_forbids", "history"),
        _list_problem(logs, final, "", "telemetry_forbids", "telemetry"),
        _list_problem(transcript, final, "", "transcript_forbids", "transcript"),
    )
    for problem in checks:
        if problem:
            return problem
    return None


def _list_problem(
    blob: str, final: dict[str, Any], contains_key: str, forbids_key: str, label: str
) -> str | None:
    for needle in final.get(contains_key, []):
        if needle not in blob:
            return f"{label} missing {needle!r}: {snippet(blob)}"
    for needle in final.get(forbids_key, []):
        if needle and needle in blob:
            return f"{label} contains forbidden text: {snippet(blob)}"
    return None


def _read_tree(directory: Path) -> str:
    if not directory.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            parts.append(_read_file(path))
    return "\n".join(parts)


def _read_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None

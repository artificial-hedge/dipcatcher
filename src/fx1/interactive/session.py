"""Append-only per-session journal for the dipcatcher chat.

Each chat is one JSONL file under ``$FX1_CONFIG_DIR/sessions`` (default
``~/.fx1/sessions``, the same root ``fx1.interactive.profiles`` uses). The
first line is a versioned header; every later line is one event. The chat
persists turns, model switches, approvals, and job transitions by appending,
and ``/sessions`` plus ``/resume`` read that log back.

On the next process start, :func:`reconcile_interrupted` settles jobs that
were still open when the process died and closes the session that never
ended, so a restart cannot treat unfinished work as finished.

This file is a mutable operational log. A job record here is **not** an
immutable research receipt: it can gain an ``interrupted`` settlement after
the fact, it is not content-addressed, and it is not evidence for a research
claim. Research evidence stays in ``receipts/`` and ``verify-research``.

Secrets never land in the journal. :func:`redact` scrubs bearer tokens, key
material, and secret-shaped blobs before a line is written.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fx1

SCHEMA_VERSION = 1

_FILE_MODE = 0o600
_DIR_MODE = 0o700
_ELLIPSIS = "…"
_TURN_MARK = "›"
_ARROW = "→"
_OMITTED_PREFIX = f"{_ELLIPSIS} (earlier lines omitted)\n"
_INTERRUPTED_DETAIL = "process exited before the job settled"
_SECRET_KEYS = frozenset({"api_key", "apikey", "authorization", "token", "secret", "password"})
_ROLE_LABELS = {"user": "you", "assistant": "dip", "system": "sys"}

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_PREFIX_RE = re.compile(r"\b(?:sk-|sk_|key-)[A-Za-z0-9_\-+/=]{8,}")
_ENV_ASSIGN_RE = re.compile(
    r"\b(?:MOONSHOT_API_KEY|FX1_API_KEY|FX1_BYOK_API_KEY|FX1_LOCAL_API_KEY"
    r"|FX1_SIGNING_KEY)=\S+",
    re.IGNORECASE,
)
_BLOB_RE = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")


@dataclass(frozen=True)
class SessionEvent:
    """One journal line after the header. ``ts`` is ISO-8601 UTC to the second."""

    seq: int
    ts: str
    kind: str
    data: dict[str, Any]


@dataclass(frozen=True)
class SessionSummary:
    """One row for ``/sessions``. Counts come from the event log, not a cache."""

    session_id: str
    started_at: str
    ended_at: str | None
    model: str
    endpoint_host: str
    turns: int
    jobs_total: int
    jobs_open: int
    last_goal: str | None
    path: Path


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _config_dir() -> Path:
    """Same rule as ``fx1.interactive.profiles.config_dir``."""
    override = os.environ.get("FX1_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fx1"


def sessions_dir() -> Path:
    """``$FX1_CONFIG_DIR/sessions`` (default ``~/.fx1/sessions``)."""
    return _config_dir() / "sessions"


def new_session_id(now: datetime | None = None) -> str:
    """UTC timestamp plus 8 hex chars, e.g. ``20261012T030405Z-a1b2c3d4``."""
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"


def _validate_session_id(session_id: str) -> str:
    if (
        not session_id
        or session_id in {".", ".."}
        or any(character in session_id for character in "/\\\x00\n\r")
    ):
        raise ValueError(f"invalid session id {session_id!r}")
    return session_id


def _mask(secret: str) -> str:
    return f"[REDACTED {_ELLIPSIS}{secret[-4:]}]"


def _mask_any(value: Any) -> str:
    if isinstance(value, str):
        return _mask(value)
    try:
        rendered = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return _mask(rendered)


def _secret_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (_BEARER_RE, _PREFIX_RE, _ENV_ASSIGN_RE):
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    for match in _BLOB_RE.finditer(text):
        token = match.group()
        if any(character.isdigit() for character in token):
            spans.append((match.start(), match.end()))
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _redact_text(text: str) -> str:
    spans = _secret_spans(text)
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        pieces.append(_mask(text[start:end]))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def redact(value: Any) -> Any:
    """Return *value* with secret-shaped strings replaced.

    Dict keys named ``api_key``, ``apikey``, ``authorization``, ``token``,
    ``secret``, or ``password`` (any case) are replaced wholesale. Other
    strings lose bearer tokens, ``sk-`` / ``sk_`` / ``key-`` tokens, inline
    ``MOONSHOT_API_KEY=...`` / ``FX1_*_KEY=...`` assignments, and digit-bearing
    runs of 32 or more token characters. The replacement keeps the last four
    characters: ``[REDACTED …wxyz]``. Ordinary prose, URLs without credentials,
    and short ids pass through unchanged.
    """
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _SECRET_KEYS:
                redacted[key] = _mask_any(item)
            else:
                redacted[key] = redact(item)
        return redacted
    return value


def _redacted_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    redacted = redact(dict(data))
    stored: dict[str, Any] = {}
    if not isinstance(redacted, dict):
        return stored
    for key, value in redacted.items():
        if isinstance(key, str):
            stored[key] = value
    return stored


def _resolve_code_revision(explicit: str | None) -> str:
    if explicit is not None:
        return _redact_text(explicit)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            timeout=2,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return "unknown"
        stdout = completed.stdout
        if not isinstance(stdout, str):
            return "unknown"
        revision = stdout.strip()
    except Exception:
        return "unknown"
    if not revision or any(character.isspace() for character in revision):
        return "unknown"
    return _redact_text(revision)


def _dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    if os.name == "posix":
        os.chmod(directory, _DIR_MODE)


def _protect_file(path: Path) -> None:
    if os.name == "posix":
        os.chmod(path, _FILE_MODE)


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()


def _as_object(parsed: object) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    result: dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(key, str):
            result[key] = value
    return result


def _as_event(raw: dict[str, Any]) -> SessionEvent | None:
    seq = raw.get("seq")
    ts = raw.get("ts")
    kind = raw.get("kind")
    data = raw.get("data")
    if isinstance(seq, bool) or not isinstance(seq, int):
        return None
    if not isinstance(ts, str) or not isinstance(kind, str):
        return None
    payload = _as_object(data)
    if payload is None:
        return None
    return SessionEvent(seq=seq, ts=ts, kind=kind, data=payload)


def _is_header(raw: dict[str, Any]) -> bool:
    return "schema_version" in raw and "seq" not in raw


def _parse_text(text: str) -> tuple[dict[str, Any] | None, list[SessionEvent], list[str]]:
    header: dict[str, Any] | None = None
    events: list[SessionEvent] = []
    errors: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed: object = json.loads(stripped)
        except json.JSONDecodeError:
            errors.append(f"line {lineno}: not JSON")
            continue
        raw = _as_object(parsed)
        if raw is None:
            errors.append(f"line {lineno}: not an object")
            continue
        if _is_header(raw):
            if header is None:
                header = raw
            continue
        event = _as_event(raw)
        if event is None:
            errors.append(f"line {lineno}: not an event")
            continue
        events.append(event)
    return header, events, errors


def _read_path(path: Path) -> tuple[dict[str, Any] | None, list[SessionEvent], list[str]] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    return _parse_text(text)


def _session_files(directory: Path) -> list[Path]:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    files: list[Path] = []
    for path in entries:
        try:
            is_file = path.is_file()
        except OSError:
            continue
        if is_file and path.suffix == ".jsonl":
            files.append(path)
    files.sort(key=lambda item: item.name)
    return files


def _coerce_directory(directory: Path | None) -> Path:
    if directory is None:
        return sessions_dir()
    return directory.expanduser()


def _string_field(raw: Mapping[str, Any], key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if isinstance(value, str):
        return value
    return default


def _open_job_ids(events: list[SessionEvent]) -> list[str]:
    """Job ids whose latest state is ``job_started`` (a later finish closes the id)."""
    open_ids: list[str] = []
    for event in events:
        if event.kind not in {"job_started", "job_finished"}:
            continue
        job_id = event.data.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            continue
        if event.kind == "job_started":
            if job_id not in open_ids:
                open_ids.append(job_id)
        elif job_id in open_ids:
            open_ids.remove(job_id)
    return open_ids


def _has_session_end(events: list[SessionEvent]) -> bool:
    return any(event.kind == "session_end" for event in events)


def _excluded(path: Path, header: dict[str, Any] | None, exclude_session_id: str | None) -> bool:
    if exclude_session_id is None:
        return False
    if path.stem == exclude_session_id:
        return True
    return header is not None and header.get("session_id") == exclude_session_id


def _append_payloads(path: Path, payloads: list[dict[str, Any]]) -> None:
    prefix = ""
    try:
        existing = path.read_bytes()
    except OSError:
        existing = b""
    if existing and not existing.endswith(b"\n"):
        prefix = "\n"
    _append_text(path, prefix + "".join(_dump(payload) for payload in payloads))
    _protect_file(path)


def _settle_open_jobs(path: Path, events: list[SessionEvent], open_jobs: list[str]) -> None:
    last_seq = max((event.seq for event in events), default=0)
    moment = _now_iso()
    payloads: list[dict[str, Any]] = []
    for job_id in open_jobs:
        last_seq += 1
        payloads.append(
            {
                "seq": last_seq,
                "ts": moment,
                "kind": "job_finished",
                "data": {
                    "job_id": job_id,
                    "status": "interrupted",
                    "detail": _INTERRUPTED_DETAIL,
                    "artifacts": [],
                },
            }
        )
    last_seq += 1
    payloads.append(
        {
            "seq": last_seq,
            "ts": moment,
            "kind": "session_end",
            "data": {"reason": "interrupted"},
        }
    )
    _append_payloads(path, payloads)


class SessionJournal:
    """One JSONL file per session: ``<sessions_dir>/<session_id>.jsonl``.

    The directory is mode 0700 and the file is mode 0600. The first line is a
    header (``schema_version``, ``session_id``, ``started_at``, ``model``,
    ``endpoint_host``, ``code_revision``, ``fx1_version``). Later lines are
    ``{"seq", "ts", "kind", "data"}``. ``code_revision`` is a short git rev, or
    ``unknown`` when git cannot answer.

    A job line in this file is a mutable status record, not an immutable
    research receipt.
    """

    write_errors: list[str]

    def __init__(
        self,
        session_id: str | None = None,
        *,
        sessions_dir: Path | None = None,
        model: str = "fx1",
        endpoint_host: str = "unconfigured",
        code_revision: str | None = None,
    ) -> None:
        self._session_id = (
            new_session_id() if session_id is None else _validate_session_id(session_id)
        )
        directory = sessions_dir if sessions_dir is not None else _default_sessions_dir()
        self._directory = directory.expanduser()
        self._path = self._directory / f"{self._session_id}.jsonl"
        self._lock = threading.Lock()
        self._seq = 0
        self.write_errors = []
        header = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self._session_id,
            "started_at": _now_iso(),
            "model": _redact_text(model),
            "endpoint_host": _redact_text(endpoint_host),
            "code_revision": _resolve_code_revision(code_revision),
            "fx1_version": fx1.__version__,
        }
        with self._lock:
            _ensure_dir(self._directory)
            _append_text(self._path, _dump(header))
            _protect_file(self._path)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def path(self) -> Path:
        return self._path

    def record(self, kind: str, **data: Any) -> SessionEvent:
        """Append one event. Disk errors are recorded on ``write_errors`` and not raised."""
        stored = _redacted_mapping(data)
        with self._lock:
            self._seq += 1
            event = SessionEvent(seq=self._seq, ts=_now_iso(), kind=kind, data=stored)
            line = _dump({"seq": event.seq, "ts": event.ts, "kind": event.kind, "data": event.data})
            try:
                _append_text(self._path, line)
                _protect_file(self._path)
            except OSError as exc:
                self.write_errors.append(f"failed to append event {event.seq}: {exc}")
            return event

    def events(self) -> list[SessionEvent]:
        """Re-read the file. The header is skipped; a torn line is not raised."""
        with self._lock:
            parsed = _read_path(self._path)
        if parsed is None:
            return []
        return parsed[1]

    def read_errors(self) -> list[str]:
        """Parse problems in the file, including a torn or invalid trailing line."""
        with self._lock:
            parsed = _read_path(self._path)
        if parsed is None:
            return [f"unreadable: {self._path}"]
        return parsed[2]

    def turn(self, role: Literal["user", "assistant", "system"], text: str) -> SessionEvent:
        return self.record("turn", role=role, text=text)

    def model_switch(self, old: str, new: str, endpoint_host: str) -> SessionEvent:
        return self.record("model_switch", old=old, new=new, endpoint_host=endpoint_host)

    def job_started(self, job_id: str, label: str, goal: str, plan_ids: list[str]) -> SessionEvent:
        return self.record(
            "job_started",
            job_id=job_id,
            label=label,
            goal=goal,
            plan_ids=list(plan_ids),
        )

    def job_step(self, job_id: str, step: str, status: str, detail: str = "") -> SessionEvent:
        return self.record("job_step", job_id=job_id, step=step, status=status, detail=detail)

    def job_finished(
        self,
        job_id: str,
        status: Literal["done", "failed", "cancelled", "interrupted"],
        detail: str = "",
        artifacts: list[str] | None = None,
    ) -> SessionEvent:
        return self.record(
            "job_finished",
            job_id=job_id,
            status=status,
            detail=detail,
            artifacts=list(artifacts or []),
        )

    def approval(self, kind: str, question: str, approved: bool) -> SessionEvent:
        return self.record(
            "approval",
            approval_kind=kind,
            question=question,
            approved=approved,
        )

    def close(self, reason: str = "exit") -> SessionEvent:
        return self.record("session_end", reason=reason)


def _default_sessions_dir() -> Path:
    return sessions_dir()


def _summarize(path: Path) -> SessionSummary | None:
    parsed = _read_path(path)
    if parsed is None:
        return None
    header, events, _errors = parsed
    if header is None:
        return None
    session_id = _string_field(header, "session_id", path.stem) or path.stem
    turns = 0
    jobs_total = 0
    last_goal: str | None = None
    ended_at: str | None = None
    for event in events:
        if event.kind == "turn":
            turns += 1
        elif event.kind == "job_started":
            jobs_total += 1
            goal = event.data.get("goal")
            if isinstance(goal, str):
                last_goal = goal
        elif event.kind == "session_end":
            ended_at = event.ts
    return SessionSummary(
        session_id=session_id,
        started_at=_string_field(header, "started_at"),
        ended_at=ended_at,
        model=_string_field(header, "model"),
        endpoint_host=_string_field(header, "endpoint_host"),
        turns=turns,
        jobs_total=jobs_total,
        jobs_open=len(_open_job_ids(events)),
        last_goal=last_goal,
        path=path,
    )


def list_sessions(directory: Path | None = None, *, limit: int = 20) -> list[SessionSummary]:
    """Newest sessions first (``started_at``, then ``session_id``). Unreadable files are skipped."""
    root = _coerce_directory(directory)
    if not root.is_dir():
        return []
    summaries: list[SessionSummary] = []
    for path in _session_files(root):
        summary = _summarize(path)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda item: (item.started_at, item.session_id), reverse=True)
    if limit < 0:
        return []
    return summaries[:limit]


def _resolve_session(session_id_or_prefix: str, directory: Path | None) -> Path | None:
    _validate_session_id(session_id_or_prefix)
    root = _coerce_directory(directory)
    if not root.is_dir():
        return None
    matches = [path for path in _session_files(root) if path.stem.startswith(session_id_or_prefix)]
    if not matches:
        return None
    exact = [path for path in matches if path.stem == session_id_or_prefix]
    if len(exact) == 1:
        return exact[0]
    if len(matches) > 1:
        found = ", ".join(path.stem for path in matches)
        raise ValueError(f"ambiguous session prefix {session_id_or_prefix!r}: {found}")
    return matches[0]


def load_session(
    session_id_or_prefix: str, directory: Path | None = None
) -> list[SessionEvent] | None:
    """Events for a unique id or prefix, header excluded.

    ``None`` when nothing matches. :class:`ValueError` when more than one
    session matches and none equals the argument.
    """
    path = _resolve_session(session_id_or_prefix, directory)
    if path is None:
        return None
    parsed = _read_path(path)
    if parsed is None:
        return None
    return parsed[1]


def load_header(session_id_or_prefix: str, directory: Path | None = None) -> dict[str, Any] | None:
    """Header object for a unique id or prefix, or ``None`` when it cannot be read."""
    path = _resolve_session(session_id_or_prefix, directory)
    if path is None:
        return None
    parsed = _read_path(path)
    if parsed is None or parsed[0] is None:
        return None
    return parsed[0]


def reconcile_interrupted(
    directory: Path | None = None, *, exclude_session_id: str | None = None
) -> list[tuple[str, str]]:
    """Settle open jobs left behind by a dead process.

    For every session except *exclude_session_id* that has a ``job_started``
    without a matching ``job_finished`` and has no ``session_end``, append
    ``job_finished(status="interrupted")`` for each open job and then
    ``session_end(reason="interrupted")``. Sequence numbers continue from the
    last ``seq`` already in the file. A second call returns an empty list.
    """
    root = _coerce_directory(directory)
    if not root.is_dir():
        return []
    marked: list[tuple[str, str]] = []
    for path in _session_files(root):
        parsed = _read_path(path)
        if parsed is None:
            continue
        header, events, _errors = parsed
        if _excluded(path, header, exclude_session_id) or _has_session_end(events):
            continue
        open_jobs = _open_job_ids(events)
        if not open_jobs:
            continue
        _settle_open_jobs(path, events, open_jobs)
        marked.extend((path.stem, job_id) for job_id in open_jobs)
    return marked


def _flat(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _role_label(role: object) -> str:
    if isinstance(role, str):
        return _ROLE_LABELS.get(role, role)
    return "sys"


def _render(event: SessionEvent) -> str | None:
    data = event.data
    kind = event.kind
    if kind == "turn":
        return f"{_role_label(data.get('role'))} {_TURN_MARK} {_flat(data.get('text'))}"
    if kind == "model_switch":
        old = _flat(data.get("old"))
        new = _flat(data.get("new"))
        host = _flat(data.get("endpoint_host"))
        return f"[model] {old} {_ARROW} {new} @ {host}"
    if kind == "job_started":
        return f"[job {_flat(data.get('job_id'))}] started: {_flat(data.get('goal'))}"
    if kind == "job_step":
        base = f"[job {_flat(data.get('job_id'))}] {_flat(data.get('step'))}: {_flat(data.get('status'))}"
        detail = _flat(data.get("detail"))
        if detail:
            return f"{base} ({detail})"
        return base
    if kind == "job_finished":
        job_id = _flat(data.get("job_id"))
        status = _flat(data.get("status"))
        detail = _flat(data.get("detail"))
        return f"[job {job_id}] {status}: {detail}"
    if kind == "approval":
        approval_kind = _flat(data.get("approval_kind"))
        question = _flat(data.get("question"))
        decision = "approved" if data.get("approved") is True else "denied"
        return f"[approval] {approval_kind}: {question} ({decision})"
    if kind == "session_end":
        return f"[session] {_flat(data.get('reason'))}"
    return None


def transcript(events: list[SessionEvent], *, max_chars: int = 4000) -> str:
    """Resume view. Long transcripts keep the tail and a leading omission marker."""
    rendered: list[str] = []
    for event in events:
        line = _render(event)
        if line is not None:
            rendered.append(line)
    body = "\n".join(rendered)
    if len(body) <= max_chars:
        return body
    if max_chars <= len(_OMITTED_PREFIX):
        return _OMITTED_PREFIX
    tail = body[-(max_chars - len(_OMITTED_PREFIX)) :]
    if len(body) > len(tail) and body[-len(tail) - 1] != "\n":
        newline_at = tail.find("\n")
        if newline_at != -1:
            tail = tail[newline_at + 1 :]
    return _OMITTED_PREFIX + tail


__all__ = [
    "SCHEMA_VERSION",
    "SessionEvent",
    "SessionJournal",
    "SessionSummary",
    "list_sessions",
    "load_header",
    "load_session",
    "new_session_id",
    "reconcile_interrupted",
    "redact",
    "sessions_dir",
    "transcript",
]

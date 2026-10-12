"""Durable chat sessions for the dipcatcher harness.

One JSON file per session, under ``$FX1_SESSION_DIR`` or ``~/.fx1/sessions``.
Records use schema version 1 and are written atomically with mode 0600.
Credential-like fields are removed, and secret-shaped strings are redacted,
before a record is written or returned.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard, cast

SESSION_SCHEMA = 1

_MODELS = frozenset({"fx1", "fx1-lite"})
_JOB_STATES = frozenset({"idle", "running", "interrupted", "finished"})
_JOB_RESUMES = frozenset({"manual", "none"})
_SECRET_MARKERS = ("api_key", "authorization", "password", "secret", "token", "key")
_TOP_LEVEL_RAW = frozenset({"id", "created_at", "updated_at"})
_SESSION_ID = re.compile(r"^[a-f0-9]{32}$")
_MODE = 0o600
_REDACTED = "[redacted]"


class SessionError(ValueError):
    """A session record could not be stored or read."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_directory() -> Path:
    override = os.environ.get("FX1_SESSION_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fx1" / "sessions"


def _valid_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _SESSION_ID.fullmatch(value) is not None


def _schema_ok(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, int):
        return False
    return value == SESSION_SCHEMA


def _name_is_secret(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _environment_secrets() -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for name, value in os.environ.items():
        if not _name_is_secret(name):
            continue
        if not value or value in seen:
            continue
        seen.add(value)
        found.append(value)
    return tuple(found)


def _secret_in_value(value: str, secret: str) -> bool:
    if not secret:
        return False
    if len(secret) > 8:
        return secret in value
    return value == secret


def _contains_env_secret(value: str, secrets: tuple[str, ...]) -> bool:
    return any(_secret_in_value(value, secret) for secret in secrets)


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    if value.startswith("sk-") and len(value) > 8:
        return _REDACTED
    if _contains_env_secret(value, secrets):
        return _REDACTED
    return value


def _keep_literal(key: str, item: Any, *, top_level: bool) -> bool:
    if not top_level:
        return False
    if isinstance(item, dict | list):
        return False
    if key in _TOP_LEVEL_RAW:
        return True
    if key != "model":
        return False
    return item in _MODELS


def _redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    if isinstance(value, dict):
        return _redact_mapping(value, secrets, top_level=False)
    return value


def _redact_field(key: str, item: Any, secrets: tuple[str, ...], *, top_level: bool) -> Any:
    if key == "job" and isinstance(item, dict):
        return _redact_job(item, secrets)
    if _keep_literal(key, item, top_level=top_level):
        return item
    return _redact_value(item, secrets)


def _restore_choice(
    redacted: dict[str, Any],
    raw: Mapping[Any, Any],
    field: str,
    allowed: frozenset[str],
) -> None:
    value = raw.get(field)
    if isinstance(value, str) and value in allowed:
        redacted[field] = value


def _redact_job(job: Mapping[Any, Any], secrets: tuple[str, ...]) -> dict[str, Any]:
    redacted = _redact_mapping(job, secrets, top_level=False)
    _restore_choice(redacted, job, "state", _JOB_STATES)
    _restore_choice(redacted, job, "resume", _JOB_RESUMES)
    return redacted


def _redact_mapping(
    value: Mapping[Any, Any],
    secrets: tuple[str, ...],
    *,
    top_level: bool,
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SessionError("session keys must be strings")
        if _name_is_secret(key):
            continue
        redacted[key] = _redact_field(key, item, secrets, top_level=top_level)
    return redacted


def _redact(value: Any) -> Any:
    secrets = _environment_secrets()
    if isinstance(value, dict):
        return _redact_mapping(value, secrets, top_level=True)
    return _redact_value(value, secrets)


def _as_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SessionError("session record must be an object")
    return cast("dict[str, Any]", value)


def _copy_messages(messages: list[dict[str, Any]] | None) -> list[Any]:
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise SessionError("messages must be a list")
    try:
        copied = copy.deepcopy(messages)
    except (TypeError, copy.Error) as exc:
        raise SessionError("messages are not storable") from exc
    if not isinstance(copied, list):
        raise SessionError("messages must be a list")
    for item in copied:
        if not isinstance(item, dict):
            raise SessionError("messages must be a list of objects")
    return copied


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    try:
        cloned = copy.deepcopy(message)
    except (TypeError, copy.Error) as exc:
        raise SessionError("message is not storable") from exc
    if not isinstance(cloned, dict):
        raise SessionError("message must be an object")
    return cast("dict[str, Any]", cloned)


def _require_save_record(record: object) -> str:
    if not isinstance(record, dict):
        raise SessionError("session record must be an object")
    if not _schema_ok(record.get("schema")):
        raise SessionError("session schema must be 1")
    session_id = record.get("id")
    if not _valid_id(session_id):
        raise SessionError("refusing session id")
    return session_id


def _prepare_save(record: dict[str, Any], session_id: str) -> dict[str, Any]:
    try:
        cloned = copy.deepcopy(record)
    except (TypeError, copy.Error) as exc:
        raise SessionError("session record is not storable") from exc
    prepared = _as_record(_redact(cloned))
    prepared["schema"] = SESSION_SCHEMA
    prepared["id"] = session_id
    prepared["updated_at"] = _utc_now()
    return prepared


def _interrupt_job(job: object) -> None:
    """Move a running job to interrupted. Does not resume or retry it."""
    if not isinstance(job, dict):
        return
    if job.get("state") != "running":
        return
    job["state"] = "interrupted"
    job["resume"] = "manual"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SessionError("unreadable session") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SessionError("invalid session JSON") from exc
    return _as_record(parsed)


def _looks_like_session_file(path: Path) -> bool:
    return path.is_file() and path.suffix == ".json" and not path.name.startswith(".")


def _parse_utc(value: str) -> datetime | None:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _stamp_rank(value: str) -> tuple[int, float, str]:
    parsed = _parse_utc(value)
    if parsed is None:
        return (0, 0.0, value)
    return (1, parsed.timestamp(), value)


def _record_sort_key(record: Mapping[str, Any]) -> tuple[int, float, str, str]:
    updated = record.get("updated_at")
    stamp = updated if isinstance(updated, str) else ""
    session_id = record.get("id")
    identity = session_id if isinstance(session_id, str) else ""
    rank = _stamp_rank(stamp)
    return (rank[0], rank[1], rank[2], identity)


def _ensure_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SessionError("unable to create session directory") from exc


def _dump(fd: int, payload: Mapping[str, Any]) -> None:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _dump_and_replace(fd: int, tmp_path: Path, path: Path, payload: Mapping[str, Any]) -> None:
    try:
        _dump(fd, payload)
        os.chmod(tmp_path, _MODE)
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise SessionError("unable to save session") from exc


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _chmod_0600(path: Path) -> None:
    try:
        os.chmod(path, _MODE)
    except OSError as exc:
        raise SessionError("unable to set session permissions") from exc


def _atomic_write(directory: Path, path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_directory(directory)
    fd, tmp_name = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        _dump_and_replace(fd, tmp_path, path, payload)
    finally:
        _unlink_quiet(tmp_path)
    _chmod_0600(path)


class SessionStore:
    """Filesystem store for fx1 and fx1-lite chat sessions."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            self.directory = _default_directory()
        else:
            self.directory = Path(directory).expanduser()

    def _resolved_directory(self) -> Path:
        return self.directory.resolve()

    def _path_for(self, session_id: str) -> Path:
        if not _valid_id(session_id):
            raise SessionError("refusing session id")
        directory = self._resolved_directory()
        path = directory / f"{session_id}.json"
        if path.parent != directory:
            raise SessionError("refusing session id")
        return path

    def _read_file(self, path: Path, *, expected_id: str | None) -> dict[str, Any]:
        parsed = _load_json_object(path)
        if not _schema_ok(parsed.get("schema")):
            raise SessionError("unsupported session schema")
        session_id = parsed.get("id")
        if not _valid_id(session_id):
            raise SessionError("invalid session id")
        if expected_id is not None and session_id != expected_id:
            raise SessionError("session id mismatch")
        return _as_record(_redact(parsed))

    def _candidate(self, path: Path) -> dict[str, Any] | None:
        if not _looks_like_session_file(path):
            return None
        try:
            record = self._read_file(path, expected_id=None)
        except SessionError:
            return None
        updated = record.get("updated_at")
        if not isinstance(updated, str) or not updated:
            return None
        return record

    def create(self, *, model: str, messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Create an open session and persist it."""
        if model not in _MODELS:
            raise SessionError("unknown model; choose fx1 or fx1-lite")
        now = _utc_now()
        record: dict[str, Any] = {
            "schema": SESSION_SCHEMA,
            "id": uuid.uuid4().hex,
            "model": model,
            "status": "open",
            "messages": _copy_messages(messages),
            "job": {
                "state": "idle",
                "id": "",
                "steps_completed": [],
                "resume": "none",
            },
            "created_at": now,
            "updated_at": now,
        }
        stored = _as_record(_redact(record))
        session_id = stored.get("id")
        if not _valid_id(session_id):
            raise SessionError("invalid session id")
        path = self._path_for(session_id)
        _atomic_write(path.parent, path, stored)
        return stored

    def save(self, record: dict[str, Any]) -> Path:
        """Redact and atomically replace the session file."""
        session_id = _require_save_record(record)
        path = self._path_for(session_id)
        prepared = _prepare_save(record, session_id)
        _atomic_write(path.parent, path, prepared)
        record.clear()
        record.update(prepared)
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        """Load one session. Missing, corrupt, or foreign files raise."""
        path = self._path_for(session_id)
        if not path.is_file():
            raise SessionError("missing session")
        return self._read_file(path, expected_id=session_id)

    def latest(self) -> dict[str, Any] | None:
        """Return the valid session with the newest ``updated_at``."""
        directory = self._resolved_directory()
        if not directory.is_dir():
            return None
        best: dict[str, Any] | None = None
        best_key: tuple[int, float, str, str] | None = None
        for path in directory.iterdir():
            record = self._candidate(path)
            if record is None:
                continue
            key = _record_sort_key(record)
            if best_key is None or key > best_key:
                best = record
                best_key = key
        return best

    def mark_interrupted(self, session_id: str) -> dict[str, Any]:
        """Persist a manual interruption for a running job.

        Idle and finished jobs keep their state. Completed steps are left in
        place. This method does not resume or retry the job.
        """
        record = self.load(session_id)
        _interrupt_job(record.get("job"))
        self.save(record)
        return record

    def append_turn(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        """Append one redacted message and save the session."""
        if not isinstance(message, dict):
            raise SessionError("message must be an object")
        record = self.load(session_id)
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise SessionError("session messages are not a list")
        messages.append(_copy_message(message))
        self.save(record)
        return record

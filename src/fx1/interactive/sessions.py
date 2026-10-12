"""Durable chat sessions for the dipcatcher harness.

One JSON file per session, under ``$FX1_SESSION_DIR`` or ``~/.fx1/sessions``.
Records use schema version 1 and are written atomically with mode 0600.
The directory is ``0700``. A newer schema is never overwritten; schema 0 is
migrated on read. Credential-like fields are removed, and secret-shaped
strings are redacted, before a record is written or returned.
``reserve`` holds one session exclusively so a second process is told it is busy.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard, cast

from fx1.interactive.atomicstore import (
    FileLock,
    StoreError,
    StoreLockTimeout,
    ensure_private_dir,
    quarantine,
)

SESSION_SCHEMA = 1

_MODELS = frozenset({"fx1", "fx1-lite"})
_JOB_STATES = frozenset({"idle", "running", "interrupted", "finished"})
_JOB_RESUMES = frozenset({"manual", "none"})
_TOP_LEVEL_RAW = frozenset({"id", "created_at", "updated_at"})
_SESSION_ID = re.compile(r"^[a-f0-9]{32}$")
_SK_TOKEN = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")
_SECRET_NAME_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])"
    r"(?:api[_-]?key|apikey|private[_-]?key|secret|passw(?:or)?d|authorization|"
    r"credentials?|token|key)"
    r"(?:[^a-z0-9]|$)"
)
_MODE = 0o600
_LOCK_TIMEOUT_S = 10.0
_REDACTED = "[redacted]"
_HELD_PATHS = threading.local()


class SessionError(ValueError):
    """A session record could not be stored or read."""


class SessionBusy(SessionError):
    """Another writer holds this session, or this write lost the race."""


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
    """True when *name* has a credential word as its own component.

    ``SERVICE_TOKEN`` matches. ``TOKENIZERS_PARALLELISM`` does not: ``token``
    is only a prefix of a longer word, and treating it as a secret redacts
    ordinary chat text such as ``false``.
    """
    return _SECRET_NAME_RE.search(name) is not None


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
    if _contains_env_secret(value, secrets):
        return _REDACTED
    return _SK_TOKEN.sub(_REDACTED, value)


def scrub_secret_text(text: str) -> str:
    """Replace known credential values inside chat and flash text.

    The credentials file and the honesty validator stay untouched. Only
    values already present in secret-named environment variables are removed,
    and only when they are long enough to be unambiguous.
    """
    secrets = sorted(
        {item for item in _environment_secrets() if len(item) >= 8}, key=len, reverse=True
    )
    scrubbed = text
    for secret in secrets:
        if secret in scrubbed:
            scrubbed = scrubbed.replace(secret, _REDACTED)
    return _SK_TOKEN.sub(_REDACTED, scrubbed)


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
    return cloned


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
    prepared["revision"] = _as_revision(record.get("revision")) + 1
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
    except UnicodeError as exc:
        quarantine(path)
        raise SessionError("unreadable session") from exc
    except OSError as exc:
        raise SessionError("unreadable session") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        quarantine(path)
        raise SessionError("invalid session JSON") from exc
    return _as_record(parsed)


def _upgrade_session(parsed: dict[str, Any]) -> dict[str, Any]:
    """Migrate schema 0 (or a missing schema) into the current record shape."""
    schema = parsed.get("schema", 0)
    if schema == SESSION_SCHEMA:
        return parsed
    if not _older_schema(schema):
        return parsed
    if not _valid_id(parsed.get("id")) or parsed.get("model") not in _MODELS:
        return parsed
    upgraded = dict(parsed)
    upgraded["schema"] = SESSION_SCHEMA
    upgraded.setdefault("status", "open")
    upgraded.setdefault("messages", [])
    if "job" not in upgraded:
        upgraded["job"] = {
            "state": "idle",
            "id": "",
            "steps_completed": [],
            "resume": "none",
        }
    return upgraded


def _older_schema(schema: object) -> bool:
    if isinstance(schema, bool) or not isinstance(schema, int):
        return False
    return 0 <= schema < SESSION_SCHEMA


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
    if directory.is_symlink():
        raise SessionError("refusing to follow symlink session directory")
    try:
        ensure_private_dir(directory)
    except (OSError, StoreError) as exc:
        raise SessionError("unable to create session directory") from exc


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


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
        _fsync_dir(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise SessionError("unable to save session") from exc


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _lock_path(session_path: Path) -> Path:
    """Lock sibling of the session directory so the directory stays json-only."""
    return session_path.parent.parent / ".fx1-session-locks" / session_path.stem


@contextlib.contextmanager
def _exclusive(path: Path, *, timeout_s: float) -> Iterator[None]:
    """Hold the session lock. Re-entrant for the thread that already holds it."""
    key = str(path)
    held: set[str] = getattr(_HELD_PATHS, "paths", None) or set()
    if key in held:
        yield
        return
    lock = FileLock(_lock_path(path), timeout_s=timeout_s)
    try:
        lock.__enter__()
    except StoreLockTimeout as exc:
        raise SessionBusy("session is busy") from exc
    current = set(held)
    current.add(key)
    _HELD_PATHS.paths = current
    try:
        yield
    finally:
        current.discard(key)
        _HELD_PATHS.paths = current
        lock.__exit__(None, None, None)


def _refuse_newer_schema(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        return
    try:
        existing = _load_json_object(path)
    except SessionError:
        return
    schema = existing.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int):
        return
    if schema > SESSION_SCHEMA:
        raise SessionError("refusing to overwrite newer session schema")


def _refuse_stale(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        return
    try:
        existing = _load_json_object(path)
    except SessionError:
        return
    if _as_revision(existing.get("revision")) != _as_revision(record.get("revision")):
        raise SessionBusy("session is busy")


def _as_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


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
        if self.directory.is_symlink():
            raise SessionError("refusing to follow symlink session directory")
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
        if path.is_symlink():
            quarantine(path)
            raise SessionError("refusing to follow symlink session file")
        parsed = _upgrade_session(_load_json_object(path))
        if not _schema_ok(parsed.get("schema")):
            raise SessionError("unsupported session schema")
        session_id = parsed.get("id")
        if not _valid_id(session_id):
            raise SessionError("invalid session id")
        if expected_id is not None and session_id != expected_id:
            raise SessionError("session id mismatch")
        return _as_record(_redact(parsed))

    def _candidate(self, path: Path) -> dict[str, Any] | None:
        if path.is_symlink() or not _looks_like_session_file(path):
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
            "revision": 1,
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
        with _exclusive(path, timeout_s=_LOCK_TIMEOUT_S):
            return self._save_locked(record, session_id, path)

    def _save_locked(self, record: dict[str, Any], session_id: str, path: Path) -> Path:
        _refuse_newer_schema(path)
        _refuse_stale(path, record)
        prepared = _prepare_save(record, session_id)
        _atomic_write(path.parent, path, prepared)
        record.clear()
        record.update(prepared)
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        """Load one session. Missing, corrupt, or foreign files raise."""
        path = self._path_for(session_id)
        if not path.is_file() and not path.is_symlink():
            raise SessionError("missing session")
        return self._read_file(path, expected_id=session_id)

    @contextlib.contextmanager
    def reserve(self, session_id: str) -> Iterator[dict[str, Any]]:
        """Exclusively resume *session_id*. A second process gets ``SessionBusy``."""
        path = self._path_for(session_id)
        if not path.is_file() and not path.is_symlink():
            raise SessionError("missing session")
        with _exclusive(path, timeout_s=0.0):
            yield self.load(session_id)

    def latest(self) -> dict[str, Any] | None:
        """Return the valid session with the newest ``updated_at``."""
        directory = self._resolved_directory()
        if not directory.is_dir():
            return None
        best: dict[str, Any] | None = None
        best_key: tuple[int, float, str, str] | None = None
        for path in list(directory.iterdir()):
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
        path = self._path_for(session_id)
        with _exclusive(path, timeout_s=_LOCK_TIMEOUT_S):
            record = self.load(session_id)
            _interrupt_job(record.get("job"))
            self.save(record)
            return record

    def append_turn(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        """Append one redacted message and save the session."""
        if not isinstance(message, dict):
            raise SessionError("message must be an object")
        path = self._path_for(session_id)
        with _exclusive(path, timeout_s=_LOCK_TIMEOUT_S):
            return self._append_locked(session_id, message)

    def diagnose(self) -> tuple[str, ...]:
        """Quarantine corrupt session files and report problems. Listing does not raise."""
        directory = self._resolved_directory()
        if not directory.is_dir():
            return ()
        notes: list[str] = []
        for path in sorted(directory.iterdir()):
            note = self._diagnose_one(path)
            if note:
                notes.append(note)
        return tuple(notes)

    def _append_locked(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        record = self.load(session_id)
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise SessionError("session messages are not a list")
        messages.append(_copy_message(message))
        self.save(record)
        return record

    def _diagnose_one(self, path: Path) -> str | None:
        if path.is_symlink():
            quarantine(path)
            return f"{path.name}: refused symlink"
        if not _looks_like_session_file(path):
            return None
        try:
            self._read_file(path, expected_id=None)
        except SessionError as exc:
            return f"{path.name}: {exc}"
        return None

"""Local structured telemetry for dipcatcher sessions.

Events are appended as JSONL under ``$FX1_CONFIG_DIR/logs/<session_id>.jsonl``.
Nothing is sent over the network. Prompt, answer, and tool content are replaced
with lengths; secrets are scrubbed with :func:`fx1.interactive.redact.redact_value`.
``emit`` never raises into callers — a full disk or a permissions error increments
``dropped`` and keeps a single in-memory ``telemetry.sink_error``.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

from fx1.interactive.atomicstore import DIR_MODE, FILE_MODE, ensure_private_dir
from fx1.interactive.redact import SecretRegistry, redact_text, redact_value
from fx1.interactive.runctx import is_valid_id, require_id

EmitFn = Callable[[str, Mapping[str, Any]], None]

MAX_VALUE_CHARS = 500
_MAX_DEPTH = 8
CONTENT_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "content",
        "messages",
        "prompt",
        "answer",
        "text",
        "body",
        "arguments",
        "stdout",
        "stderr",
    }
)
EVENT_NAME_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_OMITTED_RE = re.compile(r"^\[omitted: \d+ chars\]$")
_RESERVED = frozenset({"ts", "seq", "level", "event"})
_ID_KINDS: tuple[tuple[str, str], ...] = (
    ("session_id", "s"),
    ("run_id", "r"),
    ("job_id", "j"),
)
_LEVEL_NAMES = {"off": 0, "error": 1, "info": 2, "debug": 3}


class Level(IntEnum):
    """Log threshold. ``OFF`` creates no files."""

    OFF = 0
    ERROR = 1
    INFO = 2
    DEBUG = 3


class _Unset:
    """Sentinel for values that are not JSON scalars."""


_UNSET = _Unset()

Clock = Callable[[], datetime]


def parse_level(value: str) -> Level:
    """Parse ``off|error|info|debug``. The raw text is not echoed on failure."""
    key = value.strip().lower()
    if key not in _LEVEL_NAMES:
        raise ValueError("unknown telemetry level; expected off, error, info, or debug")
    return Level(_LEVEL_NAMES[key])


def resolve_level(level: Level | str | None, environ: Mapping[str, str] | None = None) -> Level:
    """Resolve a constructor level. An unknown ``DIPCATCHER_LOG`` falls back to info."""
    if isinstance(level, Level):
        return level
    if isinstance(level, str):
        return parse_level(level)
    env = os.environ if environ is None else environ
    try:
        return parse_level(env.get("DIPCATCHER_LOG", "info"))
    except ValueError:
        return Level.INFO


def format_timestamp(moment: datetime) -> str:
    """UTC timestamp ``YYYY-MM-DDTHH:MM:SS.ffffffZ``. Naive values are read as UTC."""
    if moment.tzinfo is None:
        aware = moment.replace(tzinfo=UTC)
    else:
        aware = moment.astimezone(UTC)
    return aware.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def default_log_dir(config: Path | None = None) -> Path:
    """``<config>/logs``. Default config is ``$FX1_CONFIG_DIR`` or ``~/.fx1``."""
    if config is not None:
        return config / "logs"
    override = os.environ.get("FX1_CONFIG_DIR")
    root = Path(override).expanduser() if override else Path.home() / ".fx1"
    return root / "logs"


def sanitize_value(
    obj: Any,
    *,
    allow_content_fields: Collection[str] | None = None,
    registry: SecretRegistry | None = None,
) -> Any:
    """Content-guard, redact, and truncate *obj* so it is safe to persist.

    Content-field names are matched case-insensitively. *allow_content_fields*
    is an exact, case-sensitive escape hatch for tests. Idempotent on its own
    output: omission markers and already-truncated strings are stable.
    """
    guarded = _guard_content(obj, _allow_set(allow_content_fields))
    redacted = redact_value(guarded, registry=registry)
    return _jsonable(redacted)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _allow_set(fields: Collection[str] | None) -> frozenset[str]:
    if fields is None:
        return frozenset()
    if isinstance(fields, str):
        raise TypeError("allow_content_fields must be a collection of names, not a string")
    return frozenset(fields)


def _check_limits(max_bytes: int, max_files: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be an integer >= 1")
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
        raise ValueError("max_files must be an integer >= 1")


def _truncate(text: str, limit: int = MAX_VALUE_CHARS) -> str:
    if len(text) <= limit:
        return text
    marker = f"[truncated {len(text)} chars]"
    if len(marker) >= limit:
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def _char_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return _encoded_len(value)


def _encoded_len(value: Any) -> int:
    try:
        rendered = json.dumps(value, default=_type_name, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        return len(type(value).__name__)
    return len(rendered)


def _already_omitted(value: Any) -> bool:
    return isinstance(value, str) and _OMITTED_RE.fullmatch(value) is not None


def _guard_content(obj: Any, allow: frozenset[str], depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return "[max-depth]"
    if isinstance(obj, Mapping):
        return _guard_mapping(obj, allow, depth)
    if isinstance(obj, (list, tuple)):
        return [_guard_content(item, allow, depth + 1) for item in obj]
    return obj


def _guard_mapping(obj: Mapping[Any, Any], allow: frozenset[str], depth: int) -> dict[str, Any]:
    guarded: dict[str, Any] = {}
    for key, value in obj.items():
        name = key if isinstance(key, str) else str(key)
        guarded[name] = _guard_one(name, value, allow, depth)
    return guarded


def _guard_one(name: str, value: Any, allow: frozenset[str], depth: int) -> Any:
    blocked = name.lower() in CONTENT_FIELD_NAMES and name not in allow
    if blocked and not _already_omitted(value):
        return f"[omitted: {_char_len(value)} chars]"
    return _guard_content(value, allow, depth + 1)


def _jsonable(obj: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return "[max-depth]"
    simple = _jsonable_scalar(obj)
    if simple is not _UNSET:
        return simple
    return _jsonable_nested(obj, depth)


def _jsonable_scalar(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _truncate(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"[bytes: {len(obj)}]"
    return _UNSET


def _finite(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def _jsonable_nested(obj: Any, depth: int) -> Any:
    if isinstance(obj, Mapping):
        return _jsonable_mapping(obj, depth)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in obj]
    return _safe_repr(obj)


def _jsonable_mapping(obj: Mapping[Any, Any], depth: int) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in obj.items():
        name = key if isinstance(key, str) else str(key)
        rendered[_truncate(redact_text(name), 200)] = _jsonable(value, depth + 1)
    return rendered


def _safe_repr(obj: Any) -> str:
    try:
        text = repr(obj)
    except Exception:
        text = f"<{type(obj).__name__}>"
    if not isinstance(text, str):
        text = f"<{type(obj).__name__}>"
    return _truncate(redact_text(text))


def _valid_id(value: Any, kind: str) -> bool:
    return isinstance(value, str) and is_valid_id(value, kind)


def _clean_ids(payload: dict[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    rejected: list[str] = []
    for key, kind in _ID_KINDS:
        if _reject_id(payload, fallback, key, kind):
            rejected.append(key)
    if rejected:
        payload["id_rejected"] = rejected
    return payload


def _reject_id(payload: dict[str, Any], fallback: Mapping[str, Any], key: str, kind: str) -> bool:
    """Drop an invalid correlation id. True when a present value was unusable."""
    if key not in payload or _valid_id(payload[key], kind):
        return False
    payload.pop(key, None)
    restored = fallback.get(key)
    if _valid_id(restored, kind):
        payload[key] = restored
    return True


def _overlay(context: Mapping[str, str], fields: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(context)
    for key, value in fields.items():
        name = key if isinstance(key, str) else str(key)
        merged[name] = value
    return merged


def _prepare_body(
    context: Mapping[str, str],
    fields: Mapping[str, Any],
    allow: frozenset[str],
    registry: SecretRegistry | None,
    root_session: str,
) -> dict[str, Any]:
    cleaned = _clean_ids(_overlay(context, fields), context)
    if not _valid_id(cleaned.get("session_id"), "s"):
        cleaned["session_id"] = root_session
    safe = sanitize_value(cleaned, allow_content_fields=allow, registry=registry)
    if not isinstance(safe, dict):
        return {}
    return {key: value for key, value in safe.items() if key not in _RESERVED}


def _normalize_event(event: str, fields: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any]]:
    if isinstance(event, str) and EVENT_NAME_RE.fullmatch(event):
        payload = fields if isinstance(fields, Mapping) else {}
        return event, payload
    length = len(event) if isinstance(event, str) else 0
    return "telemetry.bad_event", {"name_len": length}


def _clock_stamp(clock: Clock) -> str:
    try:
        moment = clock()
    except Exception:
        moment = _utc_now()
    if not isinstance(moment, datetime):
        moment = _utc_now()
    return format_timestamp(moment)


def _enabled(threshold: Level, level: Level) -> bool:
    if threshold is Level.OFF or level is Level.OFF:
        return False
    return level <= threshold


def _backup_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.name}.{index}")


def _append_flags() -> int:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while len(view) > 0:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _append_line(path: Path, line: bytes) -> None:
    if path.is_symlink():
        raise OSError("refusing to follow symlink log")
    fd = os.open(path, _append_flags(), FILE_MODE)
    try:
        if os.name == "posix":
            os.chmod(path, FILE_MODE)
        _write_all(fd, line)
    finally:
        os.close(fd)


def _ensure_log_dir(directory: Path) -> None:
    if directory.is_symlink():
        raise OSError("refusing to follow symlink log directory")
    ensure_private_dir(directory)
    if os.name == "posix":
        os.chmod(directory, DIR_MODE)


def _remove_file(path: Path) -> None:
    if path.is_symlink():
        raise OSError("refusing to follow symlink log")
    if path.exists():
        path.unlink()


def _move_log(src: Path, dest: Path) -> None:
    if src.is_symlink() or dest.is_symlink():
        raise OSError("refusing to follow symlink log")
    if src.exists():
        os.replace(src, dest)


def _shift_backups(path: Path, max_files: int) -> None:
    if max_files <= 1:
        _remove_file(path)
        return
    _remove_file(_backup_path(path, max_files - 1))
    index = max_files - 2
    while index >= 1:
        _move_log(_backup_path(path, index), _backup_path(path, index + 1))
        index -= 1
    _move_log(path, _backup_path(path, 1))


def _rotate_if_needed(path: Path, incoming: int, max_bytes: int, max_files: int) -> None:
    if path.is_symlink():
        raise OSError("refusing to follow symlink log")
    if not path.exists():
        return
    size = path.stat().st_size
    if size == 0 or size + incoming <= max_bytes:
        return
    _shift_backups(path, max_files)


def _dumps_line(record: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_type_name,
    )
    return (text + "\n").encode("utf-8")


def _existing_logs(path: Path, max_files: int) -> list[Path]:
    found: list[Path] = []
    index = max_files - 1
    while index >= 1:
        backup = _backup_path(path, index)
        if backup.exists() and not backup.is_symlink():
            found.append(backup)
        index -= 1
    if path.exists() and not path.is_symlink():
        found.append(path)
    return found


def _parse_line(line: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _read_file(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _read_events(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_file(path))
    return rows


def _resanitize(
    rows: Sequence[Mapping[str, Any]],
    allow: frozenset[str],
    registry: SecretRegistry | None,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        safe = sanitize_value(dict(row), allow_content_fields=allow, registry=registry)
        if isinstance(safe, dict):
            cleaned.append(safe)
    return cleaned


def _fail_record(exc: BaseException) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ts": format_timestamp(_utc_now()),
        "seq": 0,
        "level": "error",
        "event": "telemetry.sink_error",
        "reason": type(exc).__name__,
    }
    if isinstance(exc, OSError) and isinstance(exc.errno, int):
        record["errno"] = exc.errno
    return record


@dataclass(frozen=True)
class _Prepared:
    ts: str
    level: str
    event: str
    body: dict[str, Any]


def _compose(prepared: _Prepared, seq: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ts": prepared.ts,
        "seq": seq,
        "level": prepared.level,
        "event": prepared.event,
    }
    for key in sorted(prepared.body):
        record[key] = prepared.body[key]
    return record


class _Sink:
    """Shared append state for a telemetry root and its :meth:`Telemetry.bind` children."""

    def __init__(
        self,
        *,
        directory: Path,
        filename: str,
        level: Level,
        max_bytes: int,
        max_files: int,
        clock: Clock,
        allow: frozenset[str],
        registry: SecretRegistry | None,
    ) -> None:
        self.directory = directory
        self.filename = filename
        self.session_id = filename[: -len(".jsonl")]
        self.level = level
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.clock = clock
        self.allow = allow
        self.registry = registry
        self.lock = threading.Lock()
        self.seq = 0
        self.dropped = 0
        self.sink_error: dict[str, Any] | None = None
        self.closed = False

    @property
    def path(self) -> Path:
        return self.directory / self.filename


class Telemetry:
    """Append-only redacted JSONL log for one session.

    ``bind`` returns a child that shares the file, sequence, and lock and adds
    constant context fields. Children do not open a second file.
    """

    _sink: _Sink
    _context: dict[str, str]

    def __init__(
        self,
        *,
        session_id: str,
        directory: Path | None = None,
        level: Level | str | None = None,
        max_bytes: int = 2 * 1024 * 1024,
        max_files: int = 5,
        clock: Clock | None = None,
        context: Mapping[str, str] | None = None,
        allow_content_fields: Collection[str] | None = None,
        registry: SecretRegistry | None = None,
        environ: Mapping[str, str] | None = None,
        _adopt: tuple[_Sink, dict[str, str]] | None = None,
    ) -> None:
        if _adopt is not None:
            self._sink, self._context = _adopt
            return
        checked = require_id(session_id, "s")
        _check_limits(max_bytes, max_files)
        ctx = dict(context or {})
        ctx["session_id"] = checked
        self._context = ctx
        chosen = clock or _utc_now
        self._sink = _Sink(
            directory=default_log_dir() if directory is None else directory,
            filename=f"{checked}.jsonl",
            level=resolve_level(level, environ),
            max_bytes=max_bytes,
            max_files=max_files,
            clock=chosen,
            allow=_allow_set(allow_content_fields),
            registry=registry,
        )

    def emit(
        self,
        event: str,
        fields: Mapping[str, Any] | None = None,
        *,
        level: Level = Level.INFO,
    ) -> None:
        """Append one event. Never raises; failures increment :attr:`dropped`."""
        try:
            self._emit(event, fields, level)
        except Exception as exc:
            self._fail(exc)

    def bind(self, **context: str) -> Telemetry:
        """Child sharing this sink. Context values must be strings (ids, model, family)."""
        merged = dict(self._context)
        for key, value in context.items():
            if not isinstance(value, str):
                raise TypeError("telemetry context values must be strings")
            merged[key] = value
        return Telemetry(session_id=self._context["session_id"], _adopt=(self._sink, merged))

    def as_emit(self, level: Level = Level.INFO) -> EmitFn:
        """``EmitFn`` bound to this sink at *level*."""

        def emit(event: str, fields: Mapping[str, Any]) -> None:
            self.emit(event, fields, level=level)

        return emit

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Most recent persisted events, oldest first. Already redacted."""
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            return []
        with self._sink.lock:
            rows = _read_events(_existing_logs(self._sink.path, self._sink.max_files))
        chosen = rows if n >= len(rows) else rows[-n:]
        return _resanitize(chosen, self._sink.allow, self._sink.registry)

    def log_files(self) -> list[Path]:
        """Existing log files, oldest backup first, active file last."""
        with self._sink.lock:
            return _existing_logs(self._sink.path, self._sink.max_files)

    def close(self) -> None:
        """Stop further writes. Idempotent; already-written lines stay readable."""
        with self._sink.lock:
            self._sink.closed = True

    @property
    def path(self) -> Path:
        return self._sink.path

    @property
    def directory(self) -> Path:
        return self._sink.directory

    @property
    def level(self) -> Level:
        return self._sink.level

    @property
    def dropped(self) -> int:
        return self._sink.dropped

    @property
    def sink_error(self) -> dict[str, Any] | None:
        """First sink failure, kept in memory only (not written to the log)."""
        recorded = self._sink.sink_error
        if recorded is None:
            return None
        return dict(recorded)

    def __repr__(self) -> str:
        session = self._context.get("session_id", "")
        return (
            f"Telemetry(session_id={session!r}, level={self._sink.level.name},"
            f" dropped={self._sink.dropped})"
        )

    def _emit(self, event: str, fields: Mapping[str, Any] | None, level: Level) -> None:
        if self._sink.closed or not _enabled(self._sink.level, level):
            return
        prepared = self._prepare(event, fields, level)
        with self._sink.lock:
            if self._sink.closed:
                return
            self._commit(prepared)

    def _prepare(self, event: str, fields: Mapping[str, Any] | None, level: Level) -> _Prepared:
        name, payload = _normalize_event(event, fields)
        body = _prepare_body(
            self._context,
            payload,
            self._sink.allow,
            self._sink.registry,
            self._sink.session_id,
        )
        return _Prepared(
            ts=_clock_stamp(self._sink.clock),
            level=level.name.lower(),
            event=name,
            body=body,
        )

    def _commit(self, prepared: _Prepared) -> None:
        seq = self._sink.seq + 1
        line = _dumps_line(_compose(prepared, seq))
        _ensure_log_dir(self._sink.directory)
        _rotate_if_needed(self._sink.path, len(line), self._sink.max_bytes, self._sink.max_files)
        _append_line(self._sink.path, line)
        self._sink.seq = seq

    def _fail(self, exc: BaseException) -> None:
        try:
            with self._sink.lock:
                self._sink.dropped += 1
                if self._sink.sink_error is None:
                    self._sink.sink_error = _fail_record(exc)
        except Exception:
            return


__all__ = [
    "CONTENT_FIELD_NAMES",
    "EVENT_NAME_RE",
    "MAX_VALUE_CHARS",
    "EmitFn",
    "Level",
    "Telemetry",
    "default_log_dir",
    "format_timestamp",
    "parse_level",
    "resolve_level",
    "sanitize_value",
]

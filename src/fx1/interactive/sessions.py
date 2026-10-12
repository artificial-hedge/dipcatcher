"""Durable chat sessions for the dipcatcher harness.

One JSON file per session. Directory resolution is ``$FX1_SESSION_DIR``,
then ``$FX1_CONFIG_DIR/sessions``, then ``~/.fx1/sessions``.

Records use schema version 1. Writes go through ``atomicstore`` (fsync,
replace, mode 0600, directory 0700) under a per-session ``FileLock``.
``revision`` starts at 0 when a session is created and increments on each
successful ``save``. A file whose schema is newer than this build is never
overwritten. Missing ``revision`` and ``turns`` on older files are filled in
on load and persisted on the next save.

Credential-shaped strings and secret-named fields are removed before a
record is written or returned. Numeric and boolean fields, including usage
counters such as ``max_tokens`` and ``total_tokens``, are kept.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from fx1.interactive.atomicstore import (
    FileLock,
    StoreCorruptError,
    StoreError,
    StoreLockTimeout,
    atomic_write_json,
    read_json,
)
from fx1.interactive.atomicstore import quarantine as quarantine_file
from fx1.interactive.redact import (
    DEFAULT_REGISTRY,
    SecretRegistry,
    is_sensitive_key,
    redact_text,
    redact_value,
)
from fx1.interactive.runctx import is_valid_id, require_id

SESSION_SCHEMA = 1
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_AGE_DAYS = 90
DEFAULT_LIST_LIMIT = 50
LOCK_TIMEOUT_S = 5.0
DETAIL_MAX_CHARS = 500
EXPORT_TOOL_ARGS_CHARS = 300
EXPORT_TOOL_RESULT_CHARS = 500

_MODELS = frozenset({"fx1", "fx1-lite"})
_JOB_STATES = frozenset({"idle", "running", "interrupted", "finished"})
_JOB_RESUMES = frozenset({"manual", "none"})
_SECRET_MARKERS = ("api_key", "authorization", "password", "secret", "token", "key")
_TOP_LEVEL_RAW = frozenset({"id", "created_at", "updated_at"})
_HEX_ID = re.compile(r"^[a-f0-9]{32}$")
_REDACTED = "[redacted]"

Clock = Callable[[], datetime]
Emit = Callable[[str, Mapping[str, Any]], None]
Mutate = Callable[[dict[str, Any]], None]
RecordList = list[dict[str, Any]]
StringList = list[str]


class SessionError(ValueError):
    """A session record could not be stored or read."""


class SessionConflict(SessionError):
    """The file changed after this record was loaded.

    ``found`` is the revision on disk, or ``None`` when the file is gone.
    """

    def __init__(self, session_id: str, expected: int, found: int | None) -> None:
        disk = "missing" if found is None else str(found)
        super().__init__(
            f"session {session_id} revision conflict: loaded {expected}, on disk {disk}"
        )
        self.session_id = session_id
        self.expected = expected
        self.found = found


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat()


def _parse_utc(value: str) -> datetime | None:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _default_directory() -> Path:
    override = os.environ.get("FX1_SESSION_DIR")
    if override:
        return Path(override).expanduser()
    config = os.environ.get("FX1_CONFIG_DIR")
    if config:
        return Path(config).expanduser() / "sessions"
    return Path.home() / ".fx1" / "sessions"


def _valid_session_id(value: object) -> TypeGuard[str]:
    if not isinstance(value, str) or not value:
        return False
    if "/" in value or "\\" in value or "\x00" in value:
        return False
    if value in {".", ".."}:
        return False
    if _HEX_ID.fullmatch(value):
        return True
    return is_valid_id(value, "s")


def _require_session_id(value: object) -> str:
    if not _valid_session_id(value):
        raise SessionError("refusing session id")
    if _HEX_ID.fullmatch(value):
        return value
    try:
        return require_id(value, "s")
    except ValueError as exc:
        raise SessionError("refusing session id") from exc


def _schema_state(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, int):
        return "invalid"
    if value < 0:
        return "invalid"
    if value > SESSION_SCHEMA:
        return "future"
    if value < SESSION_SCHEMA:
        return "older"
    return "current"


def _state_of(raw: Mapping[str, Any]) -> str:
    if "schema" not in raw:
        return "missing"
    return _schema_state(raw.get("schema"))


def _schema_ok(value: object) -> bool:
    return _schema_state(value) == "current"


def _stored_revision(raw: Mapping[str, Any]) -> int:
    if "revision" not in raw:
        return 0
    value = raw.get("revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SessionError("revision must be a non-negative integer")
    return value


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
    return scrubbed


def _legacy_string(value: str, secrets: tuple[str, ...]) -> str:
    if value.startswith("sk-") and len(value) > 8:
        return _REDACTED
    if _contains_env_secret(value, secrets):
        return _REDACTED
    return value


def _drop_field(key: str, item: Any) -> bool:
    if isinstance(item, int | float):
        return False
    if _name_is_secret(key):
        return True
    return is_sensitive_key(key)


def _keep_literal(key: str, item: Any, *, top_level: bool) -> bool:
    if not top_level or isinstance(item, dict | list):
        return False
    if key in _TOP_LEVEL_RAW:
        return True
    return key == "model" and item in _MODELS


def _registry_for(secrets: tuple[str, ...]) -> SecretRegistry:
    registry = SecretRegistry()
    for value in DEFAULT_REGISTRY.values():
        registry.register(value)
    for secret in secrets:
        if len(secret) > 8:
            registry.register(secret)
    return registry


def _structural_node(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _legacy_string(value, secrets)
    if isinstance(value, list):
        return [_structural_node(item, secrets) for item in value]
    if isinstance(value, dict):
        return _structural_mapping(value, secrets, top_level=False)
    return value


def _structural_mapping(
    value: Mapping[Any, Any], secrets: tuple[str, ...], *, top_level: bool
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SessionError("session keys must be strings")
        if _drop_field(key, item):
            continue
        redacted[key] = _structural_child(key, item, secrets, top_level=top_level)
    return redacted


def _structural_child(key: str, item: Any, secrets: tuple[str, ...], *, top_level: bool) -> Any:
    if _keep_literal(key, item, top_level=top_level):
        return item
    return _structural_node(item, secrets)


def _merge_dict(
    structural: Mapping[str, Any], kernel: Mapping[str, Any], *, top_level: bool
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, item in structural.items():
        if isinstance(item, str) and _keep_literal(key, item, top_level=top_level):
            merged[key] = item
            continue
        merged[key] = _merge_redacted(item, kernel.get(key, item))
    return merged


def _merge_list(structural: Sequence[Any], kernel: Any) -> list[Any]:
    kernel_list = kernel if isinstance(kernel, list) else []
    return [
        _merge_redacted(item, kernel_list[index] if index < len(kernel_list) else item)
        for index, item in enumerate(structural)
    ]


def _merge_redacted(structural: Any, kernel: Any) -> Any:
    if isinstance(structural, int | float):
        return structural
    if isinstance(structural, dict):
        kernel_dict = kernel if isinstance(kernel, dict) else {}
        return _merge_dict(structural, kernel_dict, top_level=False)
    if isinstance(structural, list):
        return _merge_list(structural, kernel)
    if isinstance(kernel, str):
        return kernel
    return structural


def _restore_choice(
    redacted: dict[str, Any],
    raw: Mapping[Any, Any],
    field: str,
    allowed: frozenset[str],
) -> None:
    value = raw.get(field)
    if isinstance(value, str) and value in allowed:
        redacted[field] = value


def _restore_job_enums(redacted: dict[str, Any], original: Mapping[str, Any]) -> None:
    job = redacted.get("job")
    source = original.get("job")
    if not isinstance(job, dict) or not isinstance(source, dict):
        return
    _restore_choice(job, source, "state", _JOB_STATES)
    _restore_choice(job, source, "resume", _JOB_RESUMES)


def _redact(value: Any) -> Any:
    secrets = _environment_secrets()
    registry = _registry_for(secrets)
    if not isinstance(value, dict):
        return _merge_redacted(
            _structural_node(value, secrets), redact_value(value, registry=registry)
        )
    structural = _structural_mapping(value, secrets, top_level=True)
    kernel = redact_value(structural, registry=registry)
    if not isinstance(kernel, dict):
        return structural
    merged = _merge_dict(structural, kernel, top_level=True)
    _restore_job_enums(merged, value)
    return merged


def _as_dict(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SessionError(message)
    out: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SessionError("session keys must be strings")
        out[key] = item
    return out


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
    return _as_dict(cloned, "message must be an object")


def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SessionError("session record must be an object")
    try:
        cloned = copy.deepcopy(record)
    except (TypeError, copy.Error) as exc:
        raise SessionError("session record is not storable") from exc
    return _as_dict(cloned, "session record must be an object")


def _require_save_record(record: object) -> str:
    if not isinstance(record, dict):
        raise SessionError("session record must be an object")
    if not _schema_ok(record.get("schema")):
        raise SessionError("session schema must be 1")
    return _require_session_id(record.get("id"))


def _interrupt_job(job: object) -> None:
    """Move a running job to interrupted. Does not resume or retry it."""
    if not isinstance(job, dict):
        return
    if job.get("state") != "running":
        return
    job["state"] = "interrupted"
    job["resume"] = "manual"


def _looks_like_session_file(path: Path) -> bool:
    return path.is_file() and path.suffix == ".json" and not path.name.startswith(".")


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


def _dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _nbytes(payload: Mapping[str, Any]) -> int:
    try:
        encoded = _dumps(payload).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SessionError("unable to save session") from exc
    return len(encoded)


def _ensure_json(payload: Mapping[str, Any]) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionError("unable to save session") from exc


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_json(payload)
    try:
        atomic_write_json(path, payload)
    except (OSError, StoreError, TypeError, ValueError) as exc:
        raise SessionError("unable to save session") from exc


def _quarantine(path: Path, enabled: bool) -> None:
    if enabled:
        quarantine_file(path)


def _read_object(path: Path, *, quarantine: bool) -> dict[str, Any]:
    try:
        parsed = read_json(path, quarantine_corrupt=quarantine)
    except FileNotFoundError as exc:
        raise SessionError("missing session") from exc
    except StoreCorruptError as exc:
        raise SessionError("invalid session JSON") from exc
    try:
        return _as_dict(parsed, "session record must be an object")
    except SessionError:
        _quarantine(path, quarantine)
        raise


def _reject_schema(raw: Mapping[str, Any], path: Path, *, quarantine: bool) -> None:
    state = _state_of(raw)
    if state == "future":
        version = raw.get("schema")
        raise SessionError(
            f"session schema {version} is newer than supported {SESSION_SCHEMA}; "
            "file left unchanged"
        )
    if state == "invalid":
        _quarantine(path, quarantine)
        raise SessionError("unsupported session schema")


def _migrate_copy(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    if _state_of(record) in {"missing", "older"}:
        record["schema"] = SESSION_SCHEMA
    record["revision"] = _stored_revision(record)
    if not isinstance(record.get("turns"), list):
        record["turns"] = []
    if not isinstance(record.get("truncated"), bool):
        record["truncated"] = False
    return record


def _check_identity(
    record: Mapping[str, Any], expected_id: str, path: Path, *, quarantine: bool
) -> None:
    session_id = record.get("id")
    if not _valid_session_id(session_id):
        _quarantine(path, quarantine)
        raise SessionError("invalid session id")
    if session_id != expected_id:
        _quarantine(path, quarantine)
        raise SessionError("session id mismatch")


def _user_spans(messages: Sequence[Any]) -> list[tuple[int, int]]:
    starts = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    spans: list[tuple[int, int]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(messages)
        spans.append((start, end))
    return spans


def _protected_indexes(messages: Sequence[Any]) -> set[int]:
    if not messages:
        return set()
    protected = {len(messages) - 1}
    spans = _user_spans(messages)
    if spans:
        start, end = spans[-1]
        protected.update(range(start, end))
    return protected


def _span_index(index: int, spans: Sequence[tuple[int, int]]) -> int | None:
    for span_index, (start, end) in enumerate(spans):
        if start <= index < end:
            return span_index
    return None


def _span_finished(
    span_index: int | None, spans: Sequence[tuple[int, int]], turns: Sequence[Any]
) -> bool:
    if span_index is None:
        return True
    if spans and span_index == len(spans) - 1:
        return False
    if span_index >= len(turns):
        return True
    turn = turns[span_index]
    if not isinstance(turn, dict):
        return True
    outcome = turn.get("outcome")
    return outcome is not None and outcome != ""


def _eligible_indexes(messages: Sequence[Any], turns: Sequence[Any]) -> list[int]:
    protected = _protected_indexes(messages)
    spans = _user_spans(messages)
    eligible: list[int] = []
    for index, message in enumerate(messages):
        if index in protected or not isinstance(message, dict):
            continue
        if _span_finished(_span_index(index, spans), spans, turns):
            eligible.append(index)
    return eligible


def _truncation_marker(content: str) -> str:
    return f"[truncated: {len(content)} chars]"


def _shrinkable(content: str) -> bool:
    if not content or content.startswith("[truncated:"):
        return False
    return len(_truncation_marker(content)) < len(content)


def _replace_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, str) or not _shrinkable(content):
        return False
    message["content"] = _truncation_marker(content)
    return True


def _shrink_pass(
    payload: Mapping[str, Any],
    messages: Sequence[Any],
    indexes: Sequence[int],
    *,
    tools: bool,
    max_bytes: int,
) -> bool:
    changed = False
    for index in indexes:
        message = messages[index]
        if not isinstance(message, dict) or (message.get("role") == "tool") != tools:
            continue
        if not _replace_content(message):
            continue
        changed = True
        if _nbytes(payload) <= max_bytes:
            break
    return changed


def _apply_size_policy(payload: dict[str, Any], max_bytes: int) -> None:
    if _nbytes(payload) <= max_bytes:
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    turns = payload.get("turns")
    turn_list = turns if isinstance(turns, list) else []
    indexes = _eligible_indexes(messages, turn_list)
    changed = _shrink_pass(payload, messages, indexes, tools=True, max_bytes=max_bytes)
    if _nbytes(payload) > max_bytes:
        again = _shrink_pass(payload, messages, indexes, tools=False, max_bytes=max_bytes)
        changed = changed or again
    if changed:
        payload["truncated"] = True


def _next_revision(session_id: str, current: int, disk: int | None, *, force: bool) -> int:
    if disk is None:
        if current != 0 and not force:
            raise SessionConflict(session_id, current, None)
        return current + 1
    if not force and disk != current:
        raise SessionConflict(session_id, current, disk)
    base = max(disk, current) if force else current
    return base + 1


def _non_negative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SessionError(f"{label} must be a non-negative integer")
    return value


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    turns = record.get("turns")
    revision = record.get("revision")
    return {
        "id": record.get("id"),
        "model": record.get("model"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "revision": revision if isinstance(revision, int) and not isinstance(revision, bool) else 0,
        "n_messages": len(messages) if isinstance(messages, list) else 0,
        "n_turns": len(turns) if isinstance(turns, list) else 0,
        "truncated": record.get("truncated") is True,
    }


def _public_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    revision = record.get("revision")
    fields: dict[str, Any] = {
        "id": record.get("id") if isinstance(record.get("id"), str) else "",
        "revision": revision if isinstance(revision, int) and not isinstance(revision, bool) else 0,
        "truncated": record.get("truncated") is True,
    }
    model = record.get("model")
    if model in _MODELS:
        fields["model"] = model
    messages = record.get("messages")
    turns = record.get("turns")
    if isinstance(messages, list):
        fields["n_messages"] = len(messages)
    if isinstance(turns, list):
        fields["n_turns"] = len(turns)
    return fields


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _prune_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    max_sessions: int,
    max_age_days: int,
) -> list[str]:
    cutoff = now.timestamp() - (max_age_days * 86400)
    aged: list[str] = []
    fresh: list[str] = []
    for record in records:
        session_id = record.get("id")
        if not isinstance(session_id, str):
            continue
        updated = record.get("updated_at")
        stamp = _parse_utc(updated) if isinstance(updated, str) else None
        if stamp is not None and stamp.timestamp() < cutoff:
            aged.append(session_id)
        else:
            fresh.append(session_id)
    return _unique(aged + fresh[max_sessions:])


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated: {len(text)} chars]"


def _inline(text: str) -> str:
    return text.replace("`", "'")


def _usage_line(usage: Any) -> str:
    if not isinstance(usage, dict):
        return ""
    parts: list[str] = []
    for key in sorted(usage):
        value = usage[key]
        if isinstance(key, str) and isinstance(value, int | float) and not isinstance(value, bool):
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _header_lines(record: Mapping[str, Any]) -> list[str]:
    lines = [
        "# dipcatcher session",
        "",
        f"- id: `{record.get('id')}`",
        f"- model: {record.get('model')}",
        f"- status: {record.get('status')}",
        f"- revision: {record.get('revision', 0)}",
        f"- created_at: {record.get('created_at')}",
        f"- updated_at: {record.get('updated_at')}",
    ]
    if record.get("truncated") is True:
        lines.append("- truncated: true")
    lines.append("")
    return lines


def _job_lines(job: Any) -> list[str]:
    if not isinstance(job, dict):
        return []
    state = job.get("state") if isinstance(job.get("state"), str) else "idle"
    resume = job.get("resume") if isinstance(job.get("resume"), str) else ""
    job_id = job.get("id") if isinstance(job.get("id"), str) else ""
    raw_steps = job.get("steps_completed")
    steps = raw_steps if isinstance(raw_steps, list) else []
    step_text = ", ".join(step for step in steps if isinstance(step, str))
    return [
        "## Job",
        "",
        f"- state: {state}",
        f"- resume: {resume}",
        f"- id: {job_id}",
        f"- steps: {step_text}",
        "",
    ]


def _turn_model(turn: Mapping[Any, Any] | None, fallback: str) -> str:
    if turn is None:
        return fallback
    model = turn.get("model")
    if isinstance(model, str) and model:
        return model
    return fallback


def _turn_meta(turn: Mapping[Any, Any]) -> list[str]:
    lines: list[str] = []
    served = turn.get("served_model")
    if isinstance(served, str) and served:
        lines.append(f"- served_model: {served}")
    started = turn.get("started_at")
    if isinstance(started, str) and started:
        lines.append(f"- started_at: {started}")
    detail = turn.get("detail")
    if isinstance(detail, str) and detail:
        lines.append(f"- detail: {detail}")
    usage = _usage_line(turn.get("usage"))
    if usage:
        lines.append(f"- usage: {usage}")
    if lines:
        lines.append("")
    return lines


def _turn_header(index: int, turn: Mapping[Any, Any] | None, model: str) -> list[str]:
    outcome = "ok"
    if turn is not None and isinstance(turn.get("outcome"), str) and turn.get("outcome"):
        outcome = str(turn["outcome"])
    lines = [f"## Turn {index} · {model} · {outcome}", ""]
    if turn is not None:
        lines.extend(_turn_meta(turn))
    return lines


def _args_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return ""


def _named_call(source: Mapping[str, Any]) -> tuple[str, str]:
    name = source.get("name")
    name_text = name if isinstance(name, str) and name else "unknown"
    return name_text, _args_text(source.get("arguments"))


def _call_bits(call: Any) -> tuple[str, str]:
    if not isinstance(call, dict):
        return "unknown", ""
    function = call.get("function")
    if isinstance(function, dict):
        return _named_call(function)
    return _named_call(call)


def _one_call(call: Any) -> list[str]:
    name, args = _call_bits(call)
    safe_args = _inline(_clip(args, EXPORT_TOOL_ARGS_CHARS))
    return [f"**tool** · `{_inline(name)}`", "", f"args: `{safe_args}`", ""]


def _tool_call_lines(calls: Any) -> list[str]:
    if not isinstance(calls, list):
        return []
    lines: list[str] = []
    for call in calls:
        lines.extend(_one_call(call))
    return lines


def _role_label(role: str, assistant_model: str) -> str:
    if role == "assistant" and assistant_model:
        return f"**assistant** · {assistant_model}"
    if role == "tool":
        return "**tool result**"
    return f"**{role}**"


def _one_message(message: Mapping[str, Any], assistant_model: str) -> list[str]:
    raw_role = message.get("role")
    role = raw_role if isinstance(raw_role, str) else "unknown"
    lines = [_role_label(role, assistant_model), ""]
    content = message.get("content")
    if isinstance(content, str) and content:
        text = _clip(content, EXPORT_TOOL_RESULT_CHARS) if role == "tool" else content
        lines.extend((text, ""))
    lines.extend(_tool_call_lines(message.get("tool_calls")))
    call_id = message.get("tool_call_id")
    if role == "tool" and isinstance(call_id, str) and call_id:
        lines.extend((f"tool_call_id: `{_inline(_clip(call_id, 80))}`", ""))
    return lines


def _message_lines(messages: Sequence[Any], assistant_model: str) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            lines.extend(_one_message(message, assistant_model))
    return lines


def _turn_at(turns: Sequence[Any], index: int) -> dict[Any, Any] | None:
    if index >= len(turns):
        return None
    item: Any = turns[index]
    if not isinstance(item, dict):
        return None
    return item


def _append_span(
    lines: list[str],
    *,
    index: int,
    messages: Sequence[Any],
    turn: Mapping[Any, Any] | None,
    fallback: str,
    previous: str,
) -> str:
    current = _turn_model(turn, fallback)
    if previous and current != previous:
        lines.extend((f"> model switch: {previous} -> {current}", ""))
    lines.extend(_turn_header(index, turn, current))
    lines.extend(_message_lines(messages, current))
    return current


def _render_spans(
    lines: list[str],
    messages: Sequence[Any],
    turns: Sequence[Any],
    spans: Sequence[tuple[int, int]],
    fallback: str,
) -> str:
    start = spans[0][0]
    if start > 0:
        lines.extend(_message_lines(messages[:start], fallback))
    previous = ""
    for index, (span_start, span_end) in enumerate(spans):
        previous = _append_span(
            lines,
            index=index,
            messages=messages[span_start:span_end],
            turn=_turn_at(turns, index),
            fallback=fallback,
            previous=previous,
        )
    return previous


def _orphan_turns(
    lines: list[str],
    turns: Sequence[Any],
    *,
    start: int,
    fallback: str,
    previous: str,
) -> None:
    current = previous
    for offset, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        current = _append_span(
            lines,
            index=start + offset,
            messages=(),
            turn=turn,
            fallback=fallback,
            previous=current,
        )


def _transcript_lines(record: Mapping[str, Any]) -> list[str]:
    messages = record.get("messages")
    message_list = messages if isinstance(messages, list) else []
    turns = record.get("turns")
    turn_list = turns if isinstance(turns, list) else []
    raw_model = record.get("model")
    fallback = raw_model if isinstance(raw_model, str) else ""
    spans = _user_spans(message_list)
    lines: list[str] = []
    previous = ""
    if spans:
        previous = _render_spans(lines, message_list, turn_list, spans, fallback)
    else:
        lines.extend(_message_lines(message_list, fallback))
    _orphan_turns(
        lines,
        turn_list[len(spans) :],
        start=len(spans),
        fallback=fallback,
        previous=previous,
    )
    return lines


def _export_text(record: Mapping[str, Any]) -> str:
    lines = _header_lines(record)
    lines.extend(_job_lines(record.get("job")))
    lines.extend(_transcript_lines(record))
    return "\n".join(lines).rstrip() + "\n"


def _finish_export(text: str) -> str:
    scrubbed = scrub_secret_text(text)
    return redact_text(scrubbed, registry=_registry_for(_environment_secrets()))


def _usage_copy(usage: Mapping[str, Any] | None) -> dict[str, int | float] | None:
    if usage is None:
        return None
    if isinstance(usage, str) or not isinstance(usage, Mapping):
        raise SessionError("usage must be a mapping of numbers")
    copied: dict[str, int | float] = {}
    for key, value in usage.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            raise SessionError("usage must be a mapping of numbers")
        copied[key] = value
    return copied


def _checked_model(model: str | None) -> str:
    if not isinstance(model, str) or not model.strip():
        raise SessionError("model is required")
    return model


def _checked_outcome(outcome: str) -> str:
    if not isinstance(outcome, str):
        raise SessionError("outcome must be a string")
    return outcome


def _checked_detail(detail: str) -> str:
    if not isinstance(detail, str):
        raise SessionError("detail must be a string")
    return detail[:DETAIL_MAX_CHARS]


def _checked_served(served_model: str | None) -> str | None:
    if served_model is None:
        return None
    if not isinstance(served_model, str) or not served_model.strip():
        raise SessionError("served_model must be a non-empty string or null")
    return served_model


def _checked_started(started_at: str | None, now: str) -> str:
    if started_at is None:
        return now
    if not isinstance(started_at, str) or not started_at:
        raise SessionError("started_at must be a non-empty string")
    return started_at


def _new_turn(
    *,
    model: str,
    served_model: str | None,
    usage: dict[str, int | float] | None,
    outcome: str,
    detail: str,
    started_at: str,
) -> dict[str, Any]:
    return {
        "index": 0,
        "model": model,
        "served_model": served_model,
        "usage": usage,
        "outcome": outcome,
        "started_at": started_at,
        "detail": detail,
    }


class SessionStore:
    """Filesystem store for fx1 and fx1-lite chat sessions."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        clock: Clock | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
        emit: Emit | None = None,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise SessionError("max_bytes must be a positive integer")
        if isinstance(lock_timeout_s, bool) or not isinstance(lock_timeout_s, int | float):
            raise SessionError("lock_timeout_s must be positive")
        if lock_timeout_s <= 0:
            raise SessionError("lock_timeout_s must be positive")
        if directory is None:
            self.directory = _default_directory()
        else:
            self.directory = Path(directory).expanduser()
        self._clock: Clock = clock or _utc_now
        self._max_bytes = max_bytes
        self._lock_timeout_s = float(lock_timeout_s)
        self._emit_fn = emit

    def create(self, *, model: str, messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Create an open session and persist it at revision 0."""
        if model not in _MODELS:
            raise SessionError("unknown model; choose fx1 or fx1-lite")
        now = self._now_text()
        record: dict[str, Any] = {
            "schema": SESSION_SCHEMA,
            "id": uuid.uuid4().hex,
            "model": model,
            "status": "open",
            "messages": _copy_messages(messages),
            "job": {"state": "idle", "id": "", "steps_completed": [], "resume": "none"},
            "created_at": now,
            "updated_at": now,
            "revision": 0,
            "turns": [],
            "truncated": False,
        }
        stored = _as_dict(_redact(record), "session record must be an object")
        session_id = _require_session_id(stored.get("id"))
        prepared = self._prepare(stored, session_id, 0, stamp=False)
        path = self._path_for(session_id)
        with self._lock(path):
            if path.exists():
                raise SessionError("session already exists")
            _write_payload(path, prepared)
        self._emit("session.created", _public_fields(prepared))
        return prepared

    def save(self, record: dict[str, Any], *, force: bool = False) -> Path:
        """Redact and atomically replace the session file.

        Increments ``revision`` when the in-memory revision matches disk.
        ``force`` writes over a mismatch. A newer schema or a corrupt file is
        left unchanged. On success the caller's dict matches the file.
        """
        session_id = _require_save_record(record)
        path = self._path_for(session_id)
        cloned = _copy_record(record)
        try:
            prepared = self._commit(path, cloned, session_id, force=force)
        except SessionConflict as exc:
            self._emit(
                "session.conflict",
                {"id": exc.session_id, "expected": exc.expected, "found": exc.found},
            )
            raise
        record.clear()
        record.update(prepared)
        self._emit("session.saved", _public_fields(prepared))
        return path

    def load(self, session_id: str, *, quarantine: bool = False) -> dict[str, Any]:
        """Load one session. Missing, corrupt, or foreign files raise.

        ``quarantine`` moves a corrupt or mismatched file aside. A newer
        schema is never quarantined and never overwritten.
        """
        path = self._path_for(session_id)
        if not path.is_file():
            raise SessionError("missing session")
        return self._load_unlocked(path, expected_id=session_id, quarantine=quarantine)

    def latest(self) -> dict[str, Any] | None:
        """Return the valid session with the newest ``updated_at``."""
        records, _warnings = self._scan()
        best: dict[str, Any] | None = None
        best_key: tuple[int, float, str, str] | None = None
        for record in records:
            key = _record_sort_key(record)
            if best_key is None or key > best_key:
                best = record
                best_key = key
        return best

    def list(self, limit: int = DEFAULT_LIST_LIMIT) -> tuple[RecordList, StringList]:
        """Newest-first summaries, plus warnings for files that were skipped."""
        bounded = _non_negative(limit, "limit")
        records, warnings = self._scan()
        ordered = sorted(records, key=_record_sort_key, reverse=True)
        return [_summary(record) for record in ordered[:bounded]], warnings

    def delete(self, session_id: str) -> bool:
        """Remove the session file. False when it was already absent."""
        path = self._path_for(session_id)
        with self._lock(path):
            if not path.is_file():
                return False
            try:
                path.unlink()
            except OSError as exc:
                raise SessionError("unable to delete session") from exc
        self._emit("session.deleted", {"id": session_id})
        return True

    def prune(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        clock: Clock | None = None,
    ) -> int:
        """Delete sessions older than ``max_age_days`` and past ``max_sessions``.

        Age uses ``updated_at`` and the injected clock. Corrupt files are left
        in place. Returns how many session files were removed.
        """
        sessions = _non_negative(max_sessions, "max_sessions")
        age_days = _non_negative(max_age_days, "max_age_days")
        records, _warnings = self._scan()
        ordered = sorted(records, key=_record_sort_key, reverse=True)
        victims = _prune_ids(
            ordered,
            now=self._coerce_clock(clock),
            max_sessions=sessions,
            max_age_days=age_days,
        )
        return self._delete_ids(victims)

    def resolve_prefix(self, text: str) -> str:
        """Resolve ``last``, a full id, or a unique id prefix."""
        if not isinstance(text, str):
            raise SessionError("session reference must be a string")
        cleaned = text.strip()
        if cleaned == "last":
            return self._last_id()
        if not cleaned:
            raise SessionError("empty session reference")
        return _pick_prefix(cleaned, self._known_ids())

    def export_markdown(self, record: Mapping[str, Any]) -> str:
        """Redacted transcript. Does not mutate ``record`` or touch the disk."""
        redacted = _as_dict(_redact(_copy_record(record)), "session record must be an object")
        return _finish_export(_export_text(redacted))

    def export_json(self, record: Mapping[str, Any]) -> str:
        """Redacted session JSON. Does not mutate ``record``."""
        redacted = _as_dict(_redact(_copy_record(record)), "session record must be an object")
        return _dumps(redacted)

    def mark_interrupted(self, session_id: str) -> dict[str, Any]:
        """Persist a manual interruption for a running job.

        Idle and finished jobs keep their state. Completed steps are left in
        place. This method does not resume or retry the job.
        """

        def mutate(record: dict[str, Any]) -> None:
            _interrupt_job(record.get("job"))

        return self._update(session_id, mutate)

    def append_turn(
        self,
        session_id: str | dict[str, Any],
        message: dict[str, Any] | None = None,
        *,
        model: str | None = None,
        served_model: str | None = None,
        usage: Mapping[str, Any] | None = None,
        outcome: str = "ok",
        detail: str = "",
        started_at: str | None = None,
    ) -> dict[str, Any]:
        """Append a message, or record per-turn metadata on ``session_id``.

        ``append_turn(session_id, message)`` loads, appends one redacted
        message, and saves. ``append_turn(record, model=...)`` appends a turn
        (``index``, ``model``, ``served_model``, ``usage``, ``outcome``,
        ``started_at``, ``detail``) and saves that record.
        """
        if isinstance(session_id, dict):
            return self._record_turn(
                session_id,
                model=model,
                served_model=served_model,
                usage=usage,
                outcome=outcome,
                detail=detail,
                started_at=started_at,
            )
        if not isinstance(message, dict):
            raise SessionError("message must be an object")
        return self._append_message(session_id, message)

    def _append_message(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        cloned = _copy_message(message)

        def mutate(record: dict[str, Any]) -> None:
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise SessionError("session messages are not a list")
            messages.append(cloned)

        return self._update(session_id, mutate)

    def _record_turn(
        self,
        record: dict[str, Any],
        *,
        model: str | None,
        served_model: str | None,
        usage: Mapping[str, Any] | None,
        outcome: str,
        detail: str,
        started_at: str | None,
    ) -> dict[str, Any]:
        _require_save_record(record)
        turn = _new_turn(
            model=_checked_model(model),
            served_model=_checked_served(served_model),
            usage=_usage_copy(usage),
            outcome=_checked_outcome(outcome),
            detail=_checked_detail(detail),
            started_at=_checked_started(started_at, self._now_text()),
        )
        previous = record.get("turns")
        turns = previous if isinstance(previous, list) else []
        turn["index"] = len(turns)
        record["turns"] = [*turns, turn]
        try:
            self.save(record)
        except SessionError:
            _restore_turns(record, previous)
            raise
        return record

    def _commit(
        self, path: Path, cloned: dict[str, Any], session_id: str, *, force: bool
    ) -> dict[str, Any]:
        with self._lock(path):
            disk = _peek_revision(path)
            revision = _next_revision(session_id, _stored_revision(cloned), disk, force=force)
            prepared = self._prepare(cloned, session_id, revision, stamp=True)
            _write_payload(path, prepared)
            return prepared

    def _prepare(
        self, record: dict[str, Any], session_id: str, revision: int, *, stamp: bool
    ) -> dict[str, Any]:
        prepared = _as_dict(_redact(record), "session record must be an object")
        prepared["schema"] = SESSION_SCHEMA
        prepared["id"] = session_id
        prepared["revision"] = revision
        if stamp:
            prepared["updated_at"] = self._now_text()
        if not isinstance(prepared.get("turns"), list):
            prepared["turns"] = []
        if not isinstance(prepared.get("truncated"), bool):
            prepared["truncated"] = False
        _apply_size_policy(prepared, self._max_bytes)
        _ensure_json(prepared)
        return prepared

    def _update(self, session_id: str, mutate: Mutate) -> dict[str, Any]:
        path = self._path_for(session_id)
        with self._lock(path):
            if not path.is_file():
                raise SessionError("missing session")
            current = self._load_unlocked(path, expected_id=session_id, quarantine=False)
            revision = _stored_revision(current) + 1
            mutate(current)
            prepared = self._prepare(current, session_id, revision, stamp=True)
            _write_payload(path, prepared)
        self._emit("session.saved", _public_fields(prepared))
        return prepared

    def _load_unlocked(self, path: Path, *, expected_id: str, quarantine: bool) -> dict[str, Any]:
        raw = _read_object(path, quarantine=quarantine)
        _reject_schema(raw, path, quarantine=quarantine)
        migrated = _migrate_copy(raw)
        _check_identity(migrated, expected_id, path, quarantine=quarantine)
        return _as_dict(_redact(migrated), "session record must be an object")

    def _scan(self) -> tuple[RecordList, StringList]:
        directory = self._resolved_directory()
        if not directory.is_dir():
            return [], []
        records: RecordList = []
        warnings: StringList = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not _looks_like_session_file(path):
                continue
            record, warning = self._one_file(path)
            if warning is not None:
                warnings.append(warning)
            if record is not None:
                records.append(record)
        return records, warnings

    def _one_file(self, path: Path) -> tuple[dict[str, Any] | None, str | None]:
        if not _valid_session_id(path.stem):
            return None, f"skipped {path.name}: malformed session id"
        try:
            record = self._load_unlocked(path, expected_id=path.stem, quarantine=False)
        except SessionError as exc:
            return None, f"skipped {path.name}: {exc}"
        updated = record.get("updated_at")
        if not isinstance(updated, str) or not updated:
            return None, f"skipped {path.name}: missing updated_at"
        return record, None

    def _known_ids(self) -> StringList:
        records, _warnings = self._scan()
        return [record["id"] for record in records if isinstance(record.get("id"), str)]

    def _last_id(self) -> str:
        found = self.latest()
        if found is None or not isinstance(found.get("id"), str):
            raise SessionError("no sessions")
        return str(found["id"])

    def _delete_ids(self, victims: Sequence[str]) -> int:
        deleted = 0
        for session_id in victims:
            if self.delete(session_id):
                deleted += 1
        if deleted:
            self._emit("session.pruned", {"deleted": deleted})
        return deleted

    def _now_text(self) -> str:
        return _format_time(self._coerce_clock(None))

    def _coerce_clock(self, clock: Clock | None) -> datetime:
        current = self._clock() if clock is None else clock()
        if not isinstance(current, datetime):
            raise SessionError("clock must return a datetime")
        if current.tzinfo is None:
            return current.replace(tzinfo=UTC)
        return current.astimezone(UTC)

    def _resolved_directory(self) -> Path:
        return self.directory.resolve()

    def _path_for(self, session_id: str) -> Path:
        checked = _require_session_id(session_id)
        directory = self._resolved_directory()
        path = directory / f"{checked}.json"
        if path.parent != directory or path.name != f"{checked}.json":
            raise SessionError("refusing session id")
        return path

    @contextlib.contextmanager
    def _lock(self, path: Path) -> Iterator[None]:
        try:
            with FileLock(path, timeout_s=self._lock_timeout_s):
                yield
        except StoreLockTimeout as exc:
            raise SessionError("timed out waiting for session lock") from exc

    def _emit(self, name: str, fields: Mapping[str, Any]) -> None:
        if self._emit_fn is None:
            return
        self._emit_fn(name, dict(fields))


def _restore_turns(record: dict[str, Any], previous: Any) -> None:
    if isinstance(previous, list):
        record["turns"] = previous
        return
    record.pop("turns", None)


def _pick_prefix(cleaned: str, ids: Sequence[str]) -> str:
    exact = [session_id for session_id in ids if session_id == cleaned]
    if len(exact) == 1:
        return exact[0]
    matches = [session_id for session_id in ids if session_id.startswith(cleaned)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SessionError(f"no session matches {cleaned[:64]!r}")
    shown = ", ".join(sorted(matches)[:5])
    raise SessionError(
        f"ambiguous session prefix {cleaned[:64]!r} ({len(matches)} matches: {shown})"
    )


def _peek_revision(path: Path) -> int | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise SessionError("invalid session JSON")
    raw = _read_object(path, quarantine=False)
    _reject_schema(raw, path, quarantine=False)
    return _stored_revision(raw)

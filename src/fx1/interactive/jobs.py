"""Durable job ledger: state machine, restart reconciliation, cancel-until-confirmed.

Records live under ``$FX1_CONFIG_DIR/jobs/<job_id>.json`` (or an injected root)
and are written through :mod:`fx1.interactive.atomicstore`. Goals, argv, errors,
and receipt paths are stored redacted. Reconciliation marks unfinished work
interrupted and never retries it. A process is treated as still ours only when
its start token matches the one recorded at attach time.
"""

from __future__ import annotations

import builtins
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from fx1.interactive.atomicstore import (
    FileLock,
    StoreCorruptError,
    atomic_write_json,
    ensure_private_dir,
    quarantine,
    read_json,
    update_json,
)
from fx1.interactive.redact import redact_argv, redact_text, redact_value
from fx1.interactive.runctx import is_valid_step_id, new_job_id, new_run_id, require_id, step_id

SCHEMA_VERSION = 1

_MAX_GOAL = 4000
_MAX_DETAIL = 2000
_MAX_ARGV = 64
_MAX_ARG_CHARS = 500
_MAX_RECEIPTS = 32
_MAX_RECEIPT_CHARS = 500
_LOCK_TIMEOUT_S = 10.0
_START_INDEX = 19  # /proc/<pid>/stat field 22 (starttime), among post-comm fields

_KIND_OK = "kind must match ^[a-z][a-z0-9_-]{0,31}$"
_JOB_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "session_id",
        "run_id",
        "kind",
        "goal",
        "state",
        "created_at",
        "updated_at",
        "steps",
        "cancel_requested_at",
        "error",
        "budget",
        "revision",
    }
)
_STEP_FIELDS = frozenset(
    {
        "step_id",
        "name",
        "state",
        "attempt",
        "argv_summary",
        "idempotent",
        "idempotency_key",
        "started_at",
        "finished_at",
        "exit_code",
        "receipt_paths",
        "pid",
        "pgid",
        "pid_token",
        "detail",
    }
)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.TIMED_OUT,
        JobState.INTERRUPTED,
    }
)

_LEGAL: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED}),
    JobState.RUNNING: frozenset(
        {
            JobState.CANCELLING,
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.TIMED_OUT,
            JobState.INTERRUPTED,
        }
    ),
    JobState.CANCELLING: frozenset({JobState.CANCELLED, JobState.FAILED, JobState.INTERRUPTED}),
}


class IllegalTransition(RuntimeError):
    """The requested job or step transition is not in the state machine."""

    def __init__(
        self,
        message: str,
        *,
        current: JobState | None = None,
        target: JobState | None = None,
    ) -> None:
        super().__init__(message)
        self.current = current
        self.target = target


class JobNotFound(RuntimeError):
    """No ledger file exists for this job id."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"job not found: {job_id}")
        self.job_id = job_id


class ProcessProbe(Protocol):
    """Liveness plus a start token so a recycled pid is not treated as ours."""

    def is_alive(self, pid: int) -> bool: ...

    def token(self, pid: int) -> str | None: ...


@dataclass(frozen=True)
class ProcessNote:
    """One process observed during reconcile (pid, step, job)."""

    pid: int
    step_id: str
    job_id: str


@dataclass(frozen=True)
class ReconcileReport:
    """Startup recovery. Nothing in this report was retried."""

    interrupted: tuple[str, ...]
    orphan_running: tuple[ProcessNote, ...]
    unverifiable: tuple[ProcessNote, ...]
    corrupt: tuple[str, ...]
    terminated: tuple[ProcessNote, ...] = ()


@dataclass
class StepRecord:
    """One ledger step.

    ``replayed`` is not stored. :meth:`JobLedger.begin_step` sets it on the
    returned object when an idempotency key already succeeded anywhere in the
    ledger. Callers skip execution when it is true and do not call
    ``finish_step``. The stored attempt number is left unchanged (attempts
    start at 1).
    """

    step_id: str
    name: str
    state: JobState
    attempt: int = 1
    argv_summary: builtins.list[str] = field(default_factory=list)
    idempotent: bool = False
    idempotency_key: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    receipt_paths: builtins.list[str] = field(default_factory=list)
    pid: int | None = None
    pgid: int | None = None
    pid_token: str | None = None
    detail: str = ""
    replayed: bool = False


@dataclass
class JobRecord:
    job_id: str
    session_id: str | None
    run_id: str | None
    kind: str
    goal: str
    state: JobState
    created_at: str
    updated_at: str
    steps: builtins.list[StepRecord] = field(default_factory=list)
    cancel_requested_at: str | None = None
    error: str | None = None
    budget: dict[str, Any] | None = None
    revision: int = 1
    version: str = str(SCHEMA_VERSION)


@dataclass
class _Piece:
    interrupted_ids: builtins.list[str]
    orphans: builtins.list[ProcessNote]
    unverifiable: builtins.list[ProcessNote]
    terminated: builtins.list[ProcessNote]
    dirty: bool


@dataclass(frozen=True)
class _LinuxProc:
    present: bool
    state: str | None
    start: str | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _default_root() -> Path:
    from fx1.interactive.profiles import config_dir

    return config_dir() / "jobs"


def _clip(text: str, limit: int = _MAX_DETAIL) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _opt_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, label)


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int")
    return value


def _opt_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, label)


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive int")
    return value


def _string_key_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("expected string keys")
        out[key] = value
    return out


def _require_fields(payload: dict[str, Any], fields: frozenset[str]) -> None:
    if fields - payload.keys():
        raise ValueError("missing required fields")


def _as_state(value: Any) -> JobState:
    if not isinstance(value, str):
        raise ValueError("state must be a string")
    try:
        return JobState(value)
    except ValueError as exc:
        raise ValueError("unknown job state") from exc


def _validate_kind(kind: str) -> str:
    if not isinstance(kind, str) or not kind:
        raise ValueError(_KIND_OK)
    if not kind[0].islower() or not all(ch.isalnum() or ch in "_-" for ch in kind):
        raise ValueError(_KIND_OK)
    if len(kind) > 32:
        raise ValueError(_KIND_OK)
    return kind


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name or len(name) > 64:
        raise ValueError("step name must be 1..64 identifier characters")
    if not name[0].isalpha():
        raise ValueError("step name must be 1..64 identifier characters")
    if not all(ch.isalnum() or ch in "_.:-" for ch in name):
        raise ValueError("step name must be 1..64 identifier characters")
    return name


def _opt_id(value: Any, kind: str) -> str | None:
    if value is None:
        return None
    return require_id(_as_str(value, "id"), kind)


def _as_attempt(value: Any) -> int:
    number = _as_int(value, "attempt")
    if number < 1:
        raise ValueError("attempt must be >= 1")
    return number


def _as_revision(value: Any) -> int:
    number = _as_int(value, "revision")
    if number < 0:
        raise ValueError("revision must be >= 0")
    return number


def _opt_budget(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("budget must be an object or null")
    return _string_key_dict(value)


def _copy_budget(budget: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if budget is None:
        return None
    redacted = redact_value(dict(budget))
    try:
        encoded = json.dumps(redacted)
    except (TypeError, ValueError) as exc:
        raise ValueError("budget must be JSON-serializable") from exc
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError("budget must be a JSON object")
    return _string_key_dict(loaded)


def _summarize_argv(argv: Sequence[str]) -> builtins.list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise TypeError("argv must be a sequence of strings")
    items = list(argv)
    if len(items) > _MAX_ARGV:
        raise ValueError("argv is too long")
    _reject_bad_args(items)
    return redact_argv(items)


def _reject_bad_args(items: builtins.list[str]) -> None:
    for item in items:
        if not isinstance(item, str):
            raise TypeError("argv must be a sequence of strings")
        if len(item) > _MAX_ARG_CHARS:
            raise ValueError("argv element is too long")


def _receipts(paths: Sequence[str]) -> builtins.list[str]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("receipt_paths must be a sequence of strings")
    items = list(paths)
    if len(items) > _MAX_RECEIPTS:
        raise ValueError("too many receipt paths")
    return [_one_receipt(item) for item in items]


def _one_receipt(item: Any) -> str:
    if not isinstance(item, str):
        raise TypeError("receipt_paths must be a sequence of strings")
    return _clip(redact_text(item), _MAX_RECEIPT_CHARS)


def _string_list(value: Any, label: str) -> builtins.list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of strings")
    out: builtins.list[str] = []
    for item in value:
        out.append(_as_str(item, label))
    return out


def _clean_key(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("idempotency_key must be a string or null")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > 200 or "\n" in cleaned or "\x00" in cleaned:
        raise ValueError("idempotency_key is empty, too long, or contains a forbidden character")
    return cleaned


def _opt_token(value: Any) -> str | None:
    text = _opt_str(value, "pid_token")
    return text or None


def _transition_action(current: JobState, target: JobState) -> str:
    if current == target:
        return "noop"
    allowed = _LEGAL.get(current)
    if allowed is None or target not in allowed:
        raise IllegalTransition(
            f"cannot transition {current.value} -> {target.value}",
            current=current,
            target=target,
        )
    return "apply"


def _apply_error(record: JobRecord, state: JobState, error: str | None) -> None:
    if error is not None:
        record.error = _clip(redact_text(error))
    elif state == JobState.SUCCEEDED:
        record.error = None


def _apply_transition(record: JobRecord, state: JobState, error: str | None) -> bool:
    if _transition_action(record.state, state) == "noop":
        return False
    record.state = state
    _apply_error(record, state, error)
    return True


def _apply_cancel(record: JobRecord, now: str) -> bool:
    if record.state == JobState.CANCELLING:
        return False
    if record.state == JobState.QUEUED:
        record.state = JobState.CANCELLED
        record.cancel_requested_at = now
        return True
    if record.state != JobState.RUNNING:
        raise IllegalTransition(
            f"cannot request cancel from {record.state.value}",
            current=record.state,
            target=JobState.CANCELLING,
        )
    record.state = JobState.CANCELLING
    record.cancel_requested_at = now
    return True


def _is_resumable(step: StepRecord) -> bool:
    return step.state == JobState.INTERRUPTED and step.idempotent and not step.replayed


def _is_job_file(path: Path) -> bool:
    name = path.name
    if not name.endswith(".json") or name.startswith("."):
        return False
    return ".corrupt-" not in name


def _note(record: JobRecord, step: StepRecord) -> ProcessNote:
    return ProcessNote(pid=step.pid or 0, step_id=step.step_id, job_id=record.job_id)


def _touch(record: JobRecord, now: str) -> None:
    record.updated_at = now
    record.revision += 1


def _next_step_index(steps: builtins.list[StepRecord]) -> int:
    best = -1
    for step in steps:
        suffix = step.step_id.rsplit(".", 1)[-1]
        if suffix.isdigit():
            best = max(best, int(suffix))
    return best + 1


def _ensure_runnable(record: JobRecord) -> None:
    if record.state in TERMINAL_STATES or record.state == JobState.CANCELLING:
        raise IllegalTransition(
            f"cannot start a step while job is {record.state.value}",
            current=record.state,
        )
    if record.state == JobState.QUEUED:
        record.state = JobState.RUNNING


def _allocate_step_id(record: JobRecord, step_index: int | None, moment: datetime) -> str:
    if record.run_id is None:
        record.run_id = new_run_id(now=moment)
    index = _next_step_index(record.steps) if step_index is None else step_index
    sid = step_id(record.run_id, index)
    if any(existing.step_id == sid for existing in record.steps):
        raise IllegalTransition(f"step {sid} already exists")
    return sid


def _require_step(record: JobRecord, step_id_value: str) -> StepRecord:
    if not is_valid_step_id(step_id_value):
        raise ValueError("malformed step identifier")
    for step in record.steps:
        if step.step_id == step_id_value:
            return step
    raise IllegalTransition(f"step not found: {step_id_value}")


def _finish_in_place(
    step: StepRecord,
    state: JobState,
    exit_code: int | None,
    receipts: builtins.list[str],
    detail: str,
    now: str,
) -> bool:
    if state not in TERMINAL_STATES:
        raise IllegalTransition(
            "finish_step requires a terminal state",
            current=step.state,
            target=state,
        )
    if step.state == state:
        return False
    if step.state in TERMINAL_STATES:
        raise IllegalTransition(
            f"cannot finish step from {step.state.value} to {state.value}",
            current=step.state,
            target=state,
        )
    step.state = state
    step.exit_code = exit_code
    step.receipt_paths = list(receipts)
    step.detail = detail
    step.finished_at = now
    return True


def _safe_alive(probe: ProcessProbe, pid: int) -> bool | None:
    try:
        return bool(probe.is_alive(pid))
    except Exception:
        return None


def _safe_token(probe: ProcessProbe, pid: int) -> str | None:
    try:
        token = probe.token(pid)
    except Exception:
        return None
    if token is None:
        return None
    return str(token)


def _live_ownership(probe: ProcessProbe, step: StepRecord) -> str:
    """Return ``dead``, ``ours``, ``reused``, or ``unverifiable``."""
    if step.pid is None:
        return "dead"
    alive = _safe_alive(probe, step.pid)
    if alive is None:
        return "unverifiable"
    if not alive:
        return "dead"
    if not step.pid_token:
        return "unverifiable"
    return _token_ownership(step, _safe_token(probe, step.pid))


def _token_ownership(step: StepRecord, token: str | None) -> str:
    if token is None:
        return "unverifiable"
    if token != step.pid_token:
        return "reused"
    return "ours"


def _classify_step(step: StepRecord, probe: ProcessProbe) -> str:
    if step.state in TERMINAL_STATES:
        return "skip"
    ownership = _live_ownership(probe, step)
    if ownership == "ours":
        return "orphan"
    if ownership == "dead":
        return "dead"
    return "unverifiable"


def _mark_dead_steps(record: JobRecord, classes: builtins.list[str], now: str) -> bool:
    changed = False
    for step, kind in zip(record.steps, classes, strict=True):
        if kind != "dead" or step.state in TERMINAL_STATES:
            continue
        step.state = JobState.INTERRUPTED
        step.finished_at = now
        step.detail = "reconcile: process was not alive"
        changed = True
    return changed


def _interrupt_error(previous: JobState, classes: builtins.list[str]) -> str:
    if previous == JobState.QUEUED and not classes:
        return "reconcile: queued job had not started; marked interrupted"
    if "dead" in classes:
        return "reconcile: in-flight process was not alive; marked interrupted"
    return "reconcile: non-terminal job had no live process; marked interrupted"


def _cancel_blocker(probe: ProcessProbe, record: JobRecord) -> str | None:
    for step in record.steps:
        if step.state in TERMINAL_STATES or step.pid is None:
            continue
        reason = _ownership_block(probe, step)
        if reason is not None:
            return reason
    return None


def _ownership_block(probe: ProcessProbe, step: StepRecord) -> str | None:
    ownership = _live_ownership(probe, step)
    if ownership == "ours":
        return "cancellation not confirmed: process still running"
    if ownership == "unverifiable":
        return "cancellation not confirmed: process ownership unverifiable"
    return None


def _mark_steps_cancelled(record: JobRecord, now: str) -> None:
    record.state = JobState.CANCELLED
    for step in record.steps:
        if step.state in TERMINAL_STATES:
            continue
        step.state = JobState.CANCELLED
        step.finished_at = now
        if not step.detail:
            step.detail = "cancelled"


def _should_kill(killer: Callable[[int, int | None], bool] | None, record: JobRecord) -> bool:
    return killer is not None and record.state == JobState.CANCELLING


def _kill_step(killer: Callable[[int, int | None], bool] | None, step: StepRecord) -> bool:
    if killer is None or step.pid is None:
        return False
    try:
        return bool(killer(step.pid, step.pgid))
    except Exception:
        return False


def _parsed_created(record: JobRecord) -> datetime | None:
    try:
        return _as_utc(datetime.fromisoformat(record.created_at))
    except ValueError:
        return None


def _older_than(record: JobRecord, now: datetime, max_age_days: int) -> bool:
    created = _parsed_created(record)
    if created is None:
        return False
    return (now - created).total_seconds() >= max_age_days * 86400


def _check_prune_args(max_jobs: int, max_age_days: int) -> None:
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 0:
        raise ValueError("max_jobs must be an int >= 0")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or max_age_days < 0:
        raise ValueError("max_age_days must be an int >= 0")


def _oldest_terminal(records: builtins.list[JobRecord], count: int) -> builtins.list[JobRecord]:
    terminal = [rec for rec in records if rec.state in TERMINAL_STATES]
    terminal.sort(key=lambda rec: (rec.created_at, rec.job_id))
    return terminal[:count]


def _prune_ids(
    records: builtins.list[JobRecord], *, now: datetime, max_jobs: int, max_age_days: int
) -> builtins.list[str]:
    aged = [
        rec.job_id
        for rec in records
        if rec.state in TERMINAL_STATES and _older_than(rec, now, max_age_days)
    ]
    doomed = set(aged)
    remaining = [rec for rec in records if rec.job_id not in doomed]
    overflow = len(remaining) - max_jobs
    if overflow > 0:
        doomed.update(rec.job_id for rec in _oldest_terminal(remaining, overflow))
    return sorted(doomed)


def _recency(record: JobRecord) -> tuple[str, str]:
    return (record.updated_at, record.job_id)


def _step_to_dict(step: StepRecord) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "name": step.name,
        "state": step.state.value,
        "attempt": step.attempt,
        "argv_summary": list(step.argv_summary),
        "idempotent": step.idempotent,
        "idempotency_key": step.idempotency_key,
        "started_at": step.started_at,
        "finished_at": step.finished_at,
        "exit_code": step.exit_code,
        "receipt_paths": list(step.receipt_paths),
        "pid": step.pid,
        "pgid": step.pgid,
        "pid_token": step.pid_token,
        "detail": step.detail,
    }


def _job_to_dict(record: JobRecord) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": record.job_id,
        "session_id": record.session_id,
        "run_id": record.run_id,
        "kind": record.kind,
        "goal": record.goal,
        "state": record.state.value,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "steps": [_step_to_dict(step) for step in record.steps],
        "cancel_requested_at": record.cancel_requested_at,
        "error": record.error,
        "budget": record.budget,
        "revision": record.revision,
    }


def _step_from_dict(payload: Any) -> StepRecord:
    data = _string_key_dict(payload)
    _require_fields(data, _STEP_FIELDS)
    return StepRecord(
        step_id=_loaded_step_id(data["step_id"]),
        name=_validate_name(data["name"]),
        state=_as_state(data["state"]),
        attempt=_as_attempt(data["attempt"]),
        argv_summary=_string_list(data["argv_summary"], "argv_summary"),
        idempotent=_as_bool(data["idempotent"], "idempotent"),
        idempotency_key=_opt_str(data["idempotency_key"], "idempotency_key"),
        started_at=_opt_str(data["started_at"], "started_at"),
        finished_at=_opt_str(data["finished_at"], "finished_at"),
        exit_code=_opt_int(data["exit_code"], "exit_code"),
        receipt_paths=_string_list(data["receipt_paths"], "receipt_paths"),
        pid=_opt_int(data["pid"], "pid"),
        pgid=_opt_int(data["pgid"], "pgid"),
        pid_token=_opt_token(data["pid_token"]),
        detail=_as_str(data["detail"], "detail"),
    )


def _loaded_step_id(value: Any) -> str:
    text = _as_str(value, "step_id")
    if not is_valid_step_id(text):
        raise ValueError("malformed step identifier")
    return text


def _job_from_dict(payload: Any) -> JobRecord:
    data = _string_key_dict(payload)
    _require_fields(data, _JOB_FIELDS)
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    steps_raw = data["steps"]
    if not isinstance(steps_raw, list):
        raise ValueError("steps must be a list")
    return _job_from_fields(data, steps_raw)


def _job_from_fields(data: dict[str, Any], steps_raw: builtins.list[Any]) -> JobRecord:
    return JobRecord(
        job_id=require_id(_as_str(data["job_id"], "job_id"), "j"),
        session_id=_opt_id(data["session_id"], "s"),
        run_id=_opt_id(data["run_id"], "r"),
        kind=_validate_kind(_as_str(data["kind"], "kind")),
        goal=_as_str(data["goal"], "goal"),
        state=_as_state(data["state"]),
        created_at=_as_str(data["created_at"], "created_at"),
        updated_at=_as_str(data["updated_at"], "updated_at"),
        steps=[_step_from_dict(item) for item in steps_raw],
        cancel_requested_at=_opt_str(data["cancel_requested_at"], "cancel_requested_at"),
        error=_opt_str(data["error"], "error"),
        budget=_opt_budget(data["budget"]),
        revision=_as_revision(data["revision"]),
        version=str(SCHEMA_VERSION),
    )


def _command_env() -> dict[str, str]:
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    return env


def _run_command(argv: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=1.0,
        env=_command_env(),
    )
    return completed.returncode, completed.stdout


def _parse_stat(raw: str) -> _LinuxProc:
    end = raw.rfind(")")
    if end < 0:
        return _LinuxProc(True, None, None)
    fields = raw[end + 1 :].split()
    if len(fields) <= _START_INDEX or not fields[_START_INDEX].isdigit():
        return _LinuxProc(True, None, None)
    return _LinuxProc(True, fields[0], fields[_START_INDEX])


def _linux_proc(proc_root: Path, pid: int) -> _LinuxProc:
    path = proc_root / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _LinuxProc(False, None, None)
    except OSError:
        return _LinuxProc(True, None, None)
    return _parse_stat(raw)


def _linux_alive(proc_root: Path, pid: int) -> bool:
    info = _linux_proc(proc_root, pid)
    if not info.present:
        return False
    if info.state is None:
        return True
    return info.state not in {"Z", "X"}


def _linux_start_token(proc_root: Path, pid: int) -> str | None:
    return _linux_proc(proc_root, pid).start


def _ps_alive(runner: Callable[[Sequence[str]], tuple[int, str]], pid: int) -> bool:
    code, out = runner(["ps", "-p", str(pid), "-o", "pid="])
    return code == 0 and out.strip().isdigit()


def _ps_start_token(runner: Callable[[Sequence[str]], tuple[int, str]], pid: int) -> str | None:
    code, out = runner(["ps", "-p", str(pid), "-o", "lstart="])
    if code != 0:
        return None
    text = " ".join(out.split())
    return text or None


def _windows_alive(runner: Callable[[Sequence[str]], tuple[int, str]], pid: int) -> bool:
    """Best-effort liveness. Windows has no start token here, so ownership stays unverifiable."""
    code, out = runner(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"])
    if code != 0 or "no tasks" in out.lower():
        return False
    return f'"{pid}"' in out or f",{pid}," in out


def _signal_posix(pid: int, pgid: int | None) -> bool:
    try:
        if pgid is not None and pgid > 0 and hasattr(os, "killpg"):
            os.killpg(pgid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def terminate_process(pid: int, pgid: int | None) -> bool:
    """Send SIGTERM to an owned process. True when the signal was delivered.

    Windows has no POSIX process-group signal; this returns False there so the
    caller can inject a platform terminator. This never uses SIGKILL.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        return False
    return _signal_posix(pid, pgid)


class SystemProcessProbe:
    """Process liveness and a start token.

    Linux reads ``/proc/<pid>/stat`` field 22 (starttime in clock ticks).
    macOS and other POSIX hosts shell out to ``ps`` with an argument vector
    (no shell, one-second timeout) for a best-effort start string. Windows
    reports best-effort liveness via ``tasklist`` only; ``token`` is always
    ``None``, so a Windows step is unverifiable and will not be killed.
    """

    def __init__(
        self,
        *,
        proc_root: Path | None = None,
        runner: Callable[[Sequence[str]], tuple[int, str]] | None = None,
        platform: str | None = None,
    ) -> None:
        self._proc_root = proc_root if proc_root is not None else Path("/proc")
        self._runner = runner if runner is not None else _run_command
        self._platform = platform if platform is not None else sys.platform

    def is_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        if self._platform == "linux":
            return _linux_alive(self._proc_root, pid)
        if self._platform == "win32":
            return _windows_alive(self._runner, pid)
        return _ps_alive(self._runner, pid)

    def token(self, pid: int) -> str | None:
        if pid <= 0:
            return None
        if self._platform == "linux":
            return _linux_start_token(self._proc_root, pid)
        if self._platform == "win32":
            return None
        return _ps_start_token(self._runner, pid)


def _max_attempt(found: builtins.list[tuple[JobRecord, StepRecord]]) -> int:
    if not found:
        return 0
    return max(step.attempt for _, step in found)


def _succeeded_step(
    found: builtins.list[tuple[JobRecord, StepRecord]],
) -> tuple[JobRecord, StepRecord] | None:
    for record, step in found:
        if step.state == JobState.SUCCEEDED:
            return record, step
    return None


def _inflight_step(
    found: builtins.list[tuple[JobRecord, StepRecord]],
) -> tuple[JobRecord, StepRecord] | None:
    for record, step in found:
        if step.state not in TERMINAL_STATES:
            return record, step
    return None


class JobLedger:
    """Persistent job records with a locked read-modify-write cycle per mutation."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        probe: ProcessProbe | None = None,
        killer: Callable[[int, int | None], bool] | None = None,
    ) -> None:
        self.root = root if root is not None else _default_root()
        self._clock = clock or _utc_now
        self._probe: ProcessProbe = probe if probe is not None else SystemProcessProbe()
        self._killer = killer

    def create(
        self,
        *,
        kind: str,
        goal: str,
        session_id: str | None = None,
        run_id: str | None = None,
        budget: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        """Queue a new job. ``goal`` and ``budget`` are stored redacted."""
        checked_kind = _validate_kind(kind)
        if not isinstance(goal, str):
            raise TypeError("goal must be a string")
        if session_id is not None:
            require_id(session_id, "s")
        if run_id is not None:
            require_id(run_id, "r")
        moment = self._moment()
        now = _iso(moment)
        record = JobRecord(
            job_id=new_job_id(now=moment),
            session_id=session_id,
            run_id=run_id,
            kind=checked_kind,
            goal=_clip(redact_text(goal), _MAX_GOAL),
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
            budget=_copy_budget(budget),
            revision=1,
            version=str(SCHEMA_VERSION),
        )
        with self._ledger_lock():
            self._write_new(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        path = self._path(job_id)
        self._reject_symlink(path)
        try:
            payload = read_json(path)
        except FileNotFoundError as exc:
            raise JobNotFound(job_id) from exc
        except StoreCorruptError:
            quarantine(path)
            raise
        return self._checked_job(path, payload, job_id)

    def list(
        self, *, states: Iterable[JobState] | None = None, limit: int = 50
    ) -> builtins.list[JobRecord]:
        """Newest ``updated_at`` first. Corrupt files are quarantined and skipped."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative int")
        wanted = None if states is None else frozenset(states)
        with self._ledger_lock():
            records = self._iter_records(quarantine=True)
        if wanted is not None:
            records = [rec for rec in records if rec.state in wanted]
        records.sort(key=_recency, reverse=True)
        return records[:limit]

    def transition(self, job_id: str, state: JobState, *, error: str | None = None) -> JobRecord:
        """Apply one legal transition. Re-asserting the current state is a no-op."""

        def mutate(record: JobRecord) -> bool:
            return _apply_transition(record, state, error)

        with self._ledger_lock():
            return self._mutate(job_id, mutate)

    def begin_step(
        self,
        job_id: str,
        *,
        name: str,
        argv: Sequence[str],
        idempotent: bool,
        idempotency_key: str | None = None,
        step_index: int | None = None,
    ) -> StepRecord:
        """Start a step, or return the existing succeeded step with ``replayed`` set.

        Idempotency keys are ledger-global. An in-flight match raises
        :class:`IllegalTransition`. A terminal non-success starts another attempt.
        """
        checked_name = _validate_name(name)
        if not isinstance(idempotent, bool):
            raise TypeError("idempotent must be a bool")
        summary = _summarize_argv(argv)
        key = _clean_key(idempotency_key)
        with self._ledger_lock():
            self._load_for_step(job_id)
            replay = self._replay_or_attempt(key)
            if isinstance(replay, StepRecord):
                return replay
            return self._append_step(
                job_id,
                name=checked_name,
                summary=summary,
                key=key,
                attempt=replay,
                idempotent=idempotent,
                step_index=step_index,
            )

    def attach_process(
        self, job_id: str, step_id_value: str, *, pid: int, pgid: int | None
    ) -> None:
        """Record pid/pgid and the probe's start token (``None`` if unverifiable)."""
        checked_pid = _positive_int(pid, "pid")
        checked_pgid = None if pgid is None else _positive_int(pgid, "pgid")

        def mutate(record: JobRecord) -> bool:
            step = _require_step(record, step_id_value)
            if step.state in TERMINAL_STATES:
                raise IllegalTransition("cannot attach a process to a finished step")
            step.pid = checked_pid
            step.pgid = checked_pgid
            step.pid_token = _safe_token(self._probe, checked_pid)
            return True

        with self._ledger_lock():
            self._mutate(job_id, mutate)

    def finish_step(
        self,
        job_id: str,
        step_id_value: str,
        *,
        state: JobState,
        exit_code: int | None,
        receipt_paths: Sequence[str] = (),
        detail: str = "",
    ) -> StepRecord:
        """Mark a step terminal. Repeating the same terminal state is a no-op.

        The job's own state is left unchanged; the caller transitions it.
        """
        if not isinstance(detail, str):
            raise TypeError("detail must be a string")
        code = None if exit_code is None else _as_int(exit_code, "exit_code")
        receipts = _receipts(receipt_paths)
        detail_text = _clip(redact_text(detail))
        now = self._stamp()

        def mutate(record: JobRecord) -> bool:
            step = _require_step(record, step_id_value)
            return _finish_in_place(step, state, code, receipts, detail_text, now)

        with self._ledger_lock():
            updated = self._mutate(job_id, mutate)
        return _require_step(updated, step_id_value)

    def request_cancel(self, job_id: str) -> JobRecord:
        """``RUNNING`` becomes ``CANCELLING`` until :meth:`confirm_terminated`.

        A ``QUEUED`` job never started, so cancel completes as ``CANCELLED``.
        A second request while ``CANCELLING`` keeps the original timestamp.
        """
        now = self._stamp()

        def mutate(record: JobRecord) -> bool:
            return _apply_cancel(record, now)

        with self._ledger_lock():
            return self._mutate(job_id, mutate)

    def confirm_terminated(self, job_id: str) -> JobRecord:
        """``CANCELLING`` -> ``CANCELLED`` once no recorded process is still ours.

        An alive pid whose token matches blocks confirmation. An unverifiable
        token blocks confirmation. A mismatched token is a recycled pid and
        does not block. Already ``CANCELLED`` is a no-op.
        """
        now = self._stamp()

        def mutate(record: JobRecord) -> bool:
            return self._confirm_in_place(record, now)

        with self._ledger_lock():
            return self._mutate(job_id, mutate)

    def find_step_by_key(self, idempotency_key: str) -> tuple[JobRecord, StepRecord] | None:
        """Return the succeeded step for ``idempotency_key``, else the latest attempt."""
        key = _clean_key(idempotency_key)
        if key is None:
            return None
        with self._ledger_lock():
            found = self._steps_for_key(key)
        succeeded = _succeeded_step(found)
        if succeeded is not None:
            return succeeded
        if not found:
            return None
        return max(found, key=lambda item: (item[1].attempt, item[0].updated_at))

    def resumable_steps(self, job_id: str) -> builtins.list[StepRecord]:
        """Interrupted steps explicitly marked idempotent. Nothing else is resumed."""
        record = self.get(job_id)
        return [step for step in record.steps if _is_resumable(step)]

    def reconcile(self) -> ReconcileReport:
        """Mark crashed work interrupted. Never retries and never kills unless ``killer`` is set.

        A live process whose token matches is reported as ``orphan_running``.
        It is signalled only when the job is ``CANCELLING`` and a killer was
        injected. Token mismatches and missing tokens are ``unverifiable`` and
        are not killed. Corrupt files are quarantined.
        """
        interrupted: builtins.list[str] = []
        orphans: builtins.list[ProcessNote] = []
        unverifiable: builtins.list[ProcessNote] = []
        corrupt: builtins.list[str] = []
        terminated: builtins.list[ProcessNote] = []
        with self._ledger_lock():
            self._reconcile_all(interrupted, orphans, unverifiable, corrupt, terminated)
        return ReconcileReport(
            interrupted=tuple(interrupted),
            orphan_running=tuple(orphans),
            unverifiable=tuple(unverifiable),
            corrupt=tuple(corrupt),
            terminated=tuple(terminated),
        )

    def prune(self, *, max_jobs: int = 500, max_age_days: int = 60) -> int:
        """Delete terminal jobs past the age or count cap. Non-terminal jobs stay."""
        _check_prune_args(max_jobs, max_age_days)
        with self._ledger_lock():
            records = self._iter_records(quarantine=True)
            doomed = _prune_ids(
                records, now=self._moment(), max_jobs=max_jobs, max_age_days=max_age_days
            )
            for job_id in doomed:
                self._delete_job(job_id)
        return len(doomed)

    def _moment(self) -> datetime:
        moment = self._clock()
        if not isinstance(moment, datetime):
            raise TypeError("clock must return datetime")
        return _as_utc(moment)

    def _stamp(self) -> str:
        return _iso(self._moment())

    def _ledger_lock(self) -> FileLock:
        return FileLock(self.root / ".ledger", timeout_s=_LOCK_TIMEOUT_S)

    def _path(self, job_id: str) -> Path:
        checked = require_id(job_id, "j")
        return self.root / f"{checked}.json"

    def _write_new(self, record: JobRecord) -> None:
        ensure_private_dir(self.root)
        path = self._path(record.job_id)
        if path.exists():
            raise IllegalTransition(f"job file already exists for {record.job_id}")
        atomic_write_json(path, _job_to_dict(record))

    def _write_existing(self, record: JobRecord) -> None:
        atomic_write_json(self._path(record.job_id), _job_to_dict(record))

    def _delete_job(self, job_id: str) -> None:
        path = self._path(job_id)
        _unlink(path)
        _unlink(path.with_name(path.name + ".lock"))

    def _mutate(self, job_id: str, mutator: Callable[[JobRecord], bool]) -> JobRecord:
        path = self._path(job_id)

        def apply(current: Any) -> Any:
            record = _record_for_update(path, job_id, current)
            if not mutator(record):
                return current
            _touch(record, self._stamp())
            return _job_to_dict(record)

        try:
            updated = update_json(path, apply, lock_timeout_s=_LOCK_TIMEOUT_S)
        except FileNotFoundError as exc:
            raise JobNotFound(job_id) from exc
        if not isinstance(updated, dict):
            raise JobNotFound(job_id)
        return _job_from_dict(updated)

    def _job_files(self) -> builtins.list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(path for path in self.root.iterdir() if _is_job_file(path))

    def _read_job_file(self, path: Path, corrupt: builtins.list[str] | None) -> JobRecord | None:
        if path.is_symlink():
            _remember_corrupt(path, corrupt)
            return None
        try:
            record = _job_from_dict(read_json(path))
        except (OSError, StoreCorruptError, ValueError):
            _remember_corrupt(path, corrupt)
            return None
        if record.job_id != path.stem:
            _remember_corrupt(path, corrupt)
            return None
        return record

    def _iter_records(self, *, quarantine: bool) -> builtins.list[JobRecord]:
        corrupt: builtins.list[str] | None = [] if quarantine else None
        found: builtins.list[JobRecord] = []
        for path in self._job_files():
            record = self._read_job_file(path, corrupt)
            if record is not None:
                found.append(record)
        return found

    def _checked_job(self, path: Path, payload: Any, job_id: str) -> JobRecord:
        try:
            record = _job_from_dict(payload)
        except ValueError as exc:
            quarantine(path)
            raise StoreCorruptError(path, "schema rejected") from exc
        if record.job_id != job_id:
            quarantine(path)
            raise StoreCorruptError(path, "job id does not match file name")
        return record

    def _reject_symlink(self, path: Path) -> None:
        if path.is_symlink():
            quarantine(path)
            raise StoreCorruptError(path, "symlink job file")

    def _load_for_step(self, job_id: str) -> JobRecord:
        path = self._path(job_id)
        self._reject_symlink(path)
        try:
            payload = read_json(path)
        except FileNotFoundError as exc:
            raise JobNotFound(job_id) from exc
        return self._checked_job(path, payload, job_id)

    def _steps_for_key(self, key: str) -> builtins.list[tuple[JobRecord, StepRecord]]:
        found: builtins.list[tuple[JobRecord, StepRecord]] = []
        for record in self._iter_records(quarantine=True):
            for step in record.steps:
                if step.idempotency_key == key:
                    found.append((record, step))
        return found

    def _replay_or_attempt(self, key: str | None) -> StepRecord | int:
        if key is None:
            return 1
        found = self._steps_for_key(key)
        succeeded = _succeeded_step(found)
        if succeeded is not None:
            return replace(succeeded[1], replayed=True)
        inflight = _inflight_step(found)
        if inflight is not None:
            raise IllegalTransition(
                f"idempotency key is already in flight on job {inflight[0].job_id}"
            )
        return _max_attempt(found) + 1

    def _append_step(
        self,
        job_id: str,
        *,
        name: str,
        summary: builtins.list[str],
        key: str | None,
        attempt: int,
        idempotent: bool,
        step_index: int | None,
    ) -> StepRecord:
        moment = self._moment()
        now = _iso(moment)
        created: builtins.list[str] = []

        def mutate(record: JobRecord) -> bool:
            _ensure_runnable(record)
            sid = _allocate_step_id(record, step_index, moment)
            record.steps.append(
                StepRecord(
                    step_id=sid,
                    name=name,
                    state=JobState.RUNNING,
                    attempt=attempt,
                    argv_summary=list(summary),
                    idempotent=idempotent,
                    idempotency_key=key,
                    started_at=now,
                )
            )
            created.append(sid)
            return True

        updated = self._mutate(job_id, mutate)
        return _require_step(updated, created[0])

    def _confirm_in_place(self, record: JobRecord, now: str) -> bool:
        if record.state == JobState.CANCELLED:
            return False
        if record.state != JobState.CANCELLING:
            raise IllegalTransition(
                f"cannot confirm termination from {record.state.value}",
                current=record.state,
                target=JobState.CANCELLED,
            )
        blocker = _cancel_blocker(self._probe, record)
        if blocker is not None:
            raise IllegalTransition(blocker, current=record.state, target=JobState.CANCELLED)
        _mark_steps_cancelled(record, now)
        return True

    def _reconcile_all(
        self,
        interrupted: builtins.list[str],
        orphans: builtins.list[ProcessNote],
        unverifiable: builtins.list[ProcessNote],
        corrupt: builtins.list[str],
        terminated: builtins.list[ProcessNote],
    ) -> None:
        for path in self._job_files():
            record = self._read_job_file(path, corrupt)
            if record is None or record.state in TERMINAL_STATES:
                continue
            piece = self._reconcile_record(record)
            interrupted.extend(piece.interrupted_ids)
            orphans.extend(piece.orphans)
            unverifiable.extend(piece.unverifiable)
            terminated.extend(piece.terminated)
            if piece.dirty:
                self._write_existing(record)

    def _reconcile_record(self, record: JobRecord) -> _Piece:
        now = self._stamp()
        previous = record.state
        classes = [_classify_step(step, self._probe) for step in record.steps]
        orphans, unverifiable, terminated, kill_dirty = self._note_live(record, classes)
        dead_dirty = _mark_dead_steps(record, classes, now)
        if orphans or unverifiable:
            if kill_dirty or dead_dirty:
                _touch(record, now)
            return _Piece([], orphans, unverifiable, terminated, kill_dirty or dead_dirty)
        _touch(record, now)
        record.state = JobState.INTERRUPTED
        record.error = _interrupt_error(previous, classes)
        return _Piece([record.job_id], [], [], [], True)

    def _note_live(
        self, record: JobRecord, classes: builtins.list[str]
    ) -> tuple[
        builtins.list[ProcessNote], builtins.list[ProcessNote], builtins.list[ProcessNote], bool
    ]:
        orphans: builtins.list[ProcessNote] = []
        unverifiable: builtins.list[ProcessNote] = []
        terminated: builtins.list[ProcessNote] = []
        dirty = False
        for step, kind in zip(record.steps, classes, strict=True):
            dirty = self._note_one(record, step, kind, orphans, unverifiable, terminated) or dirty
        return orphans, unverifiable, terminated, dirty

    def _note_one(
        self,
        record: JobRecord,
        step: StepRecord,
        kind: str,
        orphans: builtins.list[ProcessNote],
        unverifiable: builtins.list[ProcessNote],
        terminated: builtins.list[ProcessNote],
    ) -> bool:
        if kind == "unverifiable":
            unverifiable.append(_note(record, step))
            return False
        if kind != "orphan":
            return False
        note = _note(record, step)
        orphans.append(note)
        if not _should_kill(self._killer, record):
            return False
        if _kill_step(self._killer, step):
            terminated.append(note)
            step.detail = "reconcile: termination requested"
            return True
        return False


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _remember_corrupt(path: Path, corrupt: builtins.list[str] | None) -> None:
    if corrupt is not None:
        corrupt.append(path.name)
    if path.exists() or path.is_symlink():
        quarantine(path)


def _record_for_update(path: Path, job_id: str, current: Any) -> JobRecord:
    if current is None:
        raise JobNotFound(job_id)
    if not isinstance(current, dict):
        raise StoreCorruptError(path, "job file must be a JSON object")
    try:
        record = _job_from_dict(current)
    except ValueError as exc:
        raise StoreCorruptError(path, "schema rejected") from exc
    if record.job_id != job_id:
        raise StoreCorruptError(path, "job id does not match file name")
    return record

"""Session dataset attachments for ``/data`` and ``dipcatcher exec --data``.

Profiles come from the registered ``score-forecasts --inspect`` harness
command. Harness stdout is untrusted. The sha256 recorded at attach is
rechecked before a run; a different file is not scored.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import re
import shlex
import stat
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fx1.interactive.datasets import attach_dataset, restore_datasets
from fx1.interactive.planner import DatasetRef
from fx1.interactive.policy import wrap_untrusted
from fx1.interactive.redact import redact_text
from fx1.interactive.terminal import sanitize_for_terminal

DATA_MAX_BYTES = 200_000_000
MAX_DATASETS = 8
MAX_PATH_CHARS = 4096
MAX_LABEL_CHARS = 128
MAX_COLUMNS = 64
MAX_COLUMN_CHARS = 128
INSPECT_TIMEOUT_S = 30
PROFILE_SCHEMA = "dataset_profile.v1"
PIT_NOTE = "Forecast issue time was not provided; point-in-time validity is NOT verified."
_USAGE = "usage: /data attach <path> | list | remove <n> | inspect <n>"
_CAP = f"dataset limit is {MAX_DATASETS}; /data remove <n> first"
_PROFILE_UNAVAILABLE = "dataset profile unavailable; attached without a profile"
_MAX_PROFILE_CHARS = 1_000_000
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_SHELL_META = re.compile(r"[;|&`$()<>]")
_READ = 1024 * 1024
_HEADER_CAP = 1024 * 1024
_MAX_COUNTED = 2**63 - 1


@dataclass(frozen=True)
class Attachment:
    """One attached file. ``path`` is the resolved regular file."""

    path: str
    kind: str
    label: str
    sha256: str
    size: int | None
    rows: int | None
    columns: tuple[str, ...]
    stale: bool
    has_time: bool
    time_monotonic: bool
    has_nulls: bool
    has_available_time: bool
    available_time_col: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "label": self.label,
            "sha256": self.sha256,
            "size": self.size,
            "rows": self.rows,
            "columns": list(self.columns),
            "stale": self.stale,
            "has_time": self.has_time,
            "time_monotonic": self.time_monotonic,
            "has_nulls": self.has_nulls,
            "has_available_time": self.has_available_time,
            "available_time_col": self.available_time_col,
        }


@dataclass(frozen=True)
class _Profile:
    rows: int
    columns: tuple[str, ...]
    has_nulls: bool
    has_time: bool
    time_monotonic: bool
    has_available_time: bool
    available_time_col: str
    sha256: str
    size: int


def handle_data(
    rest: str,
    rows: Sequence[Attachment],
    harness: Any,
    *,
    cwd: Path | None = None,
) -> tuple[list[Attachment], str]:
    """Run one ``/data`` command. The returned list replaces the session rows."""
    if "\x00" in rest:
        return list(rows), "path contains NUL"
    action, arg = _split_command(rest)
    current = list(rows)
    if action == "list":
        return current, _list_text(current)
    if action == "remove":
        return _remove(current, arg)
    if action == "inspect":
        return _inspect(current, arg, harness)
    updated, text, _ok = _attach(current, arg, harness, cwd)
    return updated, text


def attach_paths(
    raw_paths: Sequence[str],
    harness: Any,
    *,
    cwd: Path | None = None,
) -> tuple[list[Attachment], str | None]:
    """Attach every path. A refusal is one line and attaches nothing."""
    rows: list[Attachment] = []
    for raw in raw_paths:
        updated, text, ok = _attach(rows, raw, harness, cwd)
        if not ok:
            return [], _first_line(text)
        rows = updated
    return rows, None


def restore_stored(
    value: object,
    *,
    cwd: Path | None = None,
    legacy: object = None,
) -> tuple[list[Attachment], tuple[str, ...]]:
    """Rebuild attachments inside ``data/``. A changed digest is kept stale.

    An explicit empty ``attachments`` list does not fall back to legacy
    ``datasets`` rows. A missing list still does, for older sessions.
    """
    base = Path.cwd() if cwd is None else cwd
    if isinstance(value, list):
        if not value:
            return [], ()
        return _restore_list(value, base)
    return _restore_legacy(legacy, base)


def rehash_rows(rows: Sequence[Attachment]) -> tuple[list[Attachment], str | None]:
    """Re-hash before a run. A mismatch refuses the run and keeps the old digest."""
    updated: list[Attachment] = []
    reason: str | None = None
    for index, row in enumerate(rows, start=1):
        stale, missing = _digest_state(row)
        updated.append(replace(row, stale=stale))
        if stale and reason is None:
            reason = _stale_line(index, missing=missing)
    return updated, reason


def dataset_refs(rows: Sequence[Attachment]) -> tuple[DatasetRef, ...]:
    """Planner refs for files that still match the attached digest."""
    return tuple(_ref(row) for row in rows if not row.stale)


def model_context(rows: Sequence[Attachment]) -> str:
    """Labels, row counts, and column names, wrapped as untrusted data."""
    fresh = [row for row in rows if not row.stale]
    if not fresh:
        return ""
    body = "\n".join(_context_line(row) for row in fresh)
    return wrap_untrusted("dataset", "session", body, nonce="sessdata")


def annotate_summary(summary: str, *, unchecked: bool) -> str:
    """State that point-in-time validity was not checked for this score."""
    if not unchecked or _states_pit_unchecked(summary):
        return summary
    body = summary.rstrip()
    if not body:
        return PIT_NOTE
    return f"{body}\n{PIT_NOTE}"


def attachment_payload(rows: Sequence[Attachment]) -> list[dict[str, Any]]:
    return [row.as_json() for row in rows]


def merge_attachments(
    existing: Sequence[Attachment], extra: Sequence[Attachment]
) -> tuple[list[Attachment], str | None]:
    """Add ``extra``. The same resolved path replaces its previous row."""
    rows = list(existing)
    for item in extra:
        rows = _upsert(rows, item)
        if len(rows) > MAX_DATASETS:
            return list(existing), _CAP
    return rows, None


def stale_warnings(rows: Sequence[Attachment]) -> tuple[str, ...]:
    notes = [
        _stale_line(index, missing=not Path(row.path).is_file())
        for index, row in enumerate(rows, start=1)
        if row.stale
    ]
    return tuple(notes)


def file_sha256(path: Path) -> str:
    """Streaming sha256 of one regular file. A symlink or special file fails."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        return _hash_fd(descriptor)
    finally:
        os.close(descriptor)


def _split_command(rest: str) -> tuple[str, str]:
    try:
        parts = _command_tokens(rest)
    except ValueError:
        raise ValueError(_USAGE) from None
    if not parts or (len(parts) == 1 and parts[0] == "list"):
        return "list", ""
    if len(parts) == 2 and parts[0] in {"attach", "remove", "inspect"}:
        return parts[0], parts[1]
    raise ValueError(_USAGE)


def _command_tokens(rest: str) -> list[str]:
    """Split a ``/data`` line without treating ``\\`` as an escape.

    ``shlex.split`` drops backslashes, so a Windows path never reaches
    :func:`contain_path`, which is what turns them into portable separators.
    """
    lexer = shlex.shlex(rest, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""
    return list(lexer)


def _attach(
    rows: list[Attachment],
    raw: str,
    harness: Any,
    cwd: Path | None,
) -> tuple[list[Attachment], str, bool]:
    path, reason = resolve_dataset(raw, cwd=cwd)
    if path is None:
        return rows, reason or "refusing dataset", False
    if _at_cap(rows, path):
        return rows, _CAP, False
    item, text = _profile_or_attach(harness, path)
    if item is None:
        return rows, text, False
    return _upsert(rows, item), text, True


def resolve_dataset(raw: str, *, cwd: Path | None = None) -> tuple[Path | None, str | None]:
    """Resolve one dataset. Containment is :func:`attach_dataset`."""
    if "\x00" in raw:
        return None, "path contains NUL"
    base = Path.cwd() if cwd is None else cwd
    hazard = _existing_hazard(raw, base)
    if hazard is not None:
        return None, hazard
    try:
        ref = attach_dataset(raw, cwd=base)
    except ValueError as exc:
        return None, str(exc)
    path = Path(ref.path)
    problem = _file_reason(path, _lstat(path))
    if problem is not None:
        return None, problem
    return path, None


def _existing_hazard(raw: str, cwd: Path) -> str | None:
    """Refuse a named FIFO, device, or oversize file before it is read."""
    candidate = _named_file(raw, cwd)
    if candidate is None:
        return None
    info = _lstat(candidate)
    if info is None:
        return None
    if stat.S_ISLNK(info.st_mode):
        return _link_hazard(candidate)
    special = _special_reason(info)
    if special is not None:
        return special
    if info.st_size > DATA_MAX_BYTES:
        return f"dataset exceeds {DATA_MAX_BYTES} bytes"
    return None


def _named_file(raw: str, cwd: Path) -> Path | None:
    text = raw.strip()
    if not text or "\x00" in text:
        return None
    expanded = Path(text).expanduser()
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return expanded


def _link_hazard(path: Path) -> str | None:
    try:
        info = path.resolve(strict=True).lstat()
    except OSError:
        return None
    special = _special_reason(info)
    if special is not None:
        return special
    if info.st_size > DATA_MAX_BYTES:
        return f"dataset exceeds {DATA_MAX_BYTES} bytes"
    return None


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError:
        return None


def _file_reason(path: Path, info: os.stat_result | None) -> str | None:
    if info is None:
        return "dataset file was not found under data/"
    special = _special_reason(info)
    if special is not None:
        return special
    if info.st_size > DATA_MAX_BYTES:
        return f"dataset exceeds {DATA_MAX_BYTES} bytes"
    if len(str(path)) > MAX_PATH_CHARS:
        return "path is too long"
    if not os.access(path, os.R_OK):
        return "dataset is not readable"
    return None


def _special_reason(info: os.stat_result) -> str | None:
    if stat.S_ISFIFO(info.st_mode):
        return "refusing a FIFO"
    if stat.S_ISDIR(info.st_mode):
        return "refusing a directory"
    if not stat.S_ISREG(info.st_mode):
        return "refusing a non-regular file"
    return None


def _at_cap(rows: Sequence[Attachment], path: Path) -> bool:
    if any(_same_stored_path(row.path, str(path)) for row in rows):
        return False
    return len(rows) >= MAX_DATASETS


def _profile_or_attach(harness: Any, path: Path) -> tuple[Attachment | None, str]:
    head = f"attached dataset {path}"
    digest = _digest(path) or ""
    profile = _optional_profile(harness, path, digest)
    current = _digest(path) or ""
    if current != digest:
        return _after_drift(path, head, current)
    if profile is None:
        return _unprofiled(path, digest), f"{head}\n{_PROFILE_UNAVAILABLE}"
    item = _from_profile(path, profile)
    return item, f"{head}\n{_render_profile(_profile_for_display(profile, item))}"


def _profile_for_display(profile: _Profile, item: Attachment) -> _Profile:
    if item.available_time_col:
        return profile
    return replace(profile, has_available_time=False, available_time_col="")


def _after_drift(path: Path, head: str, current: str) -> tuple[Attachment | None, str]:
    if not current:
        return None, "dataset changed during attach; retry"
    item = _unprofiled(path, current)
    return item, f"{head}\nfile changed during attach; attached without a profile"


def _optional_profile(harness: Any, path: Path, digest: str) -> _Profile | None:
    if not digest:
        return None
    code, stdout = _run_inspect(harness, path)
    if _digest(path) != digest or code != 0:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return parse_profile(stdout, digest, size)


def _unprofiled(path: Path, digest: str) -> Attachment:
    kind = "parquet" if path.suffix.lower() == ".parquet" else "csv"
    size = _lstat(path)
    return Attachment(
        path=str(path),
        kind=kind,
        label="file",
        sha256=digest,
        size=None if size is None else size.st_size,
        rows=None,
        columns=(),
        stale=False,
        has_time=False,
        time_monotonic=False,
        has_nulls=False,
        has_available_time=False,
        available_time_col="",
    )


def _run_inspect(harness: Any, path: Path) -> tuple[int, str]:
    if harness is None:
        return 1, ""
    try:
        result = harness.run(
            "score-forecasts",
            ["--inspect", "--task", "volatility", "--data", str(path)],
            timeout_s=INSPECT_TIMEOUT_S,
        )
    except TypeError:
        return _run_inspect_plain(harness, path)
    except Exception:
        return 1, ""
    return _coerce_result(result)


def _run_inspect_plain(harness: Any, path: Path) -> tuple[int, str]:
    try:
        result = harness.run(
            "score-forecasts",
            ["--inspect", "--task", "volatility", "--data", str(path)],
        )
    except Exception:
        return 1, ""
    return _coerce_result(result)


def _coerce_result(result: Any) -> tuple[int, str]:
    code = getattr(result, "exit_code", 1)
    stdout = getattr(result, "stdout", "")
    if isinstance(code, bool) or not isinstance(code, int):
        return 1, ""
    if not isinstance(stdout, str) or len(stdout) > _MAX_PROFILE_CHARS:
        return 1, ""
    return code, stdout


def parse_profile(stdout: str, digest: str, size: int) -> _Profile | None:
    """Defensive ``dataset_profile.v1`` parse. ``None`` means attach without a profile."""
    payload = _json_object(stdout)
    if payload is None or not _profile_identity(payload, digest, size):
        return None
    columns = _profile_columns(payload.get("columns"))
    timing = _time_view(payload.get("time"))
    roles = payload.get("suggested_roles")
    available = _available_flag(roles)
    row_count = _as_int(payload.get("n_rows"))
    if columns is None or timing is None or available is None or row_count is None:
        return None
    names, nulls = columns
    column = _exact_issue_column(roles, names)
    has_time, monotonic = timing
    return _Profile(
        rows=row_count,
        columns=names,
        has_nulls=nulls,
        has_time=has_time,
        time_monotonic=monotonic,
        has_available_time=available,
        available_time_col=column,
        sha256=digest,
        size=size,
    )


def _json_object(stdout: str) -> dict[str, Any] | None:
    if len(stdout) > _MAX_PROFILE_CHARS:
        return None
    start = stdout.find("{")
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(stdout[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items() if isinstance(key, str)}


def _profile_identity(payload: dict[str, Any], digest: str, size: int) -> bool:
    if payload.get("schema") != PROFILE_SCHEMA:
        return False
    if not _sha_matches(payload.get("sha256"), digest):
        return False
    return payload.get("size_bytes") == size


def _sha_matches(value: object, digest: str) -> bool:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        return False
    return value.lower() == digest.lower()


def _profile_columns(value: object) -> tuple[tuple[str, ...], bool] | None:
    if not isinstance(value, list):
        return None
    names: list[str] = []
    nulls = False
    for item in value[:MAX_COLUMNS]:
        name, count = _one_column(item)
        if name is None:
            return None
        names.append(name)
        nulls = nulls or count > 0
    return tuple(names), nulls


def _one_column(item: object) -> tuple[str | None, int]:
    if not isinstance(item, dict):
        return None, 0
    raw = item.get("name")
    if not isinstance(raw, str):
        return None, 0
    count = _as_int(item.get("nulls", 0))
    if count is None:
        return None, 0
    return _clean_name(raw), count


def _time_view(value: object) -> tuple[bool, bool] | None:
    if not isinstance(value, dict):
        return None
    column = value.get("column")
    if column is not None and not isinstance(column, str):
        return None
    increasing = value.get("strictly_increasing")
    if increasing is not None and not isinstance(increasing, bool):
        return None
    has_time = isinstance(column, str) and bool(column.strip())
    return has_time, increasing is True


def _available_flag(value: object) -> bool | None:
    if not isinstance(value, dict):
        return None
    slot = value.get("available_time")
    if slot is None:
        return False
    if not isinstance(slot, dict):
        return None
    column = slot.get("column")
    if column is None:
        return False
    if not isinstance(column, str):
        return None
    return bool(column.strip())


def _exact_issue_column(roles: object, columns: tuple[str, ...]) -> str:
    raw = _issue_name(roles)
    if raw is None:
        return ""
    return _column_bindable(_clean_name(raw), columns)


def _issue_name(roles: object) -> str | None:
    if not isinstance(roles, dict):
        return None
    slot = roles.get("available_time")
    if not isinstance(slot, dict) or slot.get("confidence") != "exact_name":
        return None
    column = slot.get("column")
    if not isinstance(column, str) or not column.strip():
        return None
    return column


def _from_profile(path: Path, profile: _Profile) -> Attachment:
    kind = "parquet" if path.suffix.lower() == ".parquet" else "csv"
    names = _confirmed_columns(path, profile.columns)
    column = profile.available_time_col if profile.available_time_col in names else ""
    if _column_rejected(column):
        column = ""
    return Attachment(
        path=str(path),
        kind=kind,
        label="file",
        sha256=profile.sha256,
        size=profile.size,
        rows=profile.rows,
        columns=names,
        stale=False,
        has_time=profile.has_time,
        time_monotonic=profile.time_monotonic,
        has_nulls=profile.has_nulls,
        has_available_time=profile.has_available_time,
        available_time_col=column,
    )


def _render_profile(profile: _Profile) -> str:
    columns = ", ".join(profile.columns) or "-"
    lines = [f"profile rows={profile.rows} columns={columns}", *_warning_lines(profile)]
    return "\n".join(lines)


def _warning_lines(profile: _Profile) -> list[str]:
    notes: list[str] = []
    if not profile.has_time:
        notes.append("warning: no time column")
    elif not profile.time_monotonic:
        notes.append("warning: time is not monotonic")
    if profile.has_nulls:
        notes.append("warning: columns contain nulls")
    if not profile.has_available_time:
        notes.append("warning: point-in-time validity will NOT be verified")
    return notes


def _remove(rows: list[Attachment], token: str) -> tuple[list[Attachment], str]:
    index = _index(token, len(rows))
    if index is None:
        return rows, "unknown dataset index"
    removed = rows[index]
    kept = [row for number, row in enumerate(rows) if number != index]
    return kept, f"removed {_show_text(removed.label)}"


def _inspect(rows: list[Attachment], token: str, harness: Any) -> tuple[list[Attachment], str]:
    index = _index(token, len(rows))
    if index is None:
        return rows, "unknown dataset index"
    row = rows[index]
    text, stale = _inspect_row(row, harness)
    updated = list(rows)
    updated[index] = replace(row, stale=stale)
    return updated, text


def _inspect_row(row: Attachment, harness: Any) -> tuple[str, bool]:
    path = Path(row.path)
    reason = _stored_problem(path)
    digest = None if reason else _digest(path)
    stale = _inspect_stale(reason, digest, row.sha256)
    lines = _inspect_lines(path, harness, reason)
    if stale:
        lines.append("file changed since attach")
    return "\n".join(lines), stale


def _inspect_stale(reason: str | None, digest: str | None, previous: str) -> bool:
    if reason is not None or digest is None:
        return True
    return digest != previous


def _inspect_lines(path: Path, harness: Any, reason: str | None) -> list[str]:
    if reason is not None:
        return [reason]
    digest = _digest(path) or ""
    profile = _optional_profile(harness, path, digest)
    if profile is None:
        return [_PROFILE_UNAVAILABLE]
    return _render_profile(profile).splitlines()


def _list_text(rows: Sequence[Attachment]) -> str:
    if not rows:
        return "no datasets attached"
    return "\n".join(_list_line(index, row) for index, row in enumerate(rows, start=1))


def _list_line(index: int, row: Attachment) -> str:
    digest = row.sha256[:8] if row.sha256 else "-"
    marker = " stale" if row.stale else ""
    return (
        f"{index} {row.kind} {_show_text(Path(row.path).name)} rows={_row_count(row)} "
        f"sha={digest}{marker} {_show_text(row.path)}"
    )


def _index(token: str, count: int) -> int | None:
    if not token.isdigit():
        return None
    number = int(token)
    if number < 1 or number > count:
        return None
    return number - 1


def _upsert(rows: list[Attachment], item: Attachment) -> list[Attachment]:
    for index, row in enumerate(rows):
        if _same_stored_path(row.path, item.path):
            updated = list(rows)
            updated[index] = item
            return updated
    return [*rows, item]


def _restore_list(value: list[object], cwd: Path) -> tuple[list[Attachment], tuple[str, ...]]:
    kept: list[Attachment] = []
    warnings: list[str] = []
    overflow = False
    for item in value:
        if len(kept) >= MAX_DATASETS:
            overflow = True
            break
        _restore_one(item, kept, warnings, cwd)
    if overflow:
        warnings.append(f"dataset limit is {MAX_DATASETS}; extra records were dropped")
    return kept, tuple(warnings)


def _restore_legacy(raw: object, cwd: Path) -> tuple[list[Attachment], tuple[str, ...]]:
    kept: list[Attachment] = []
    for ref in restore_datasets(raw, cwd=cwd):
        if len(kept) >= MAX_DATASETS:
            break
        path = Path(ref.path)
        digest = _digest(path)
        if digest is None:
            continue
        info = _lstat(path)
        kept.append(
            _unprofiled(path, digest)
            if info is None
            else replace(_unprofiled(path, digest), size=info.st_size, kind=ref.kind)
        )
    return kept, ()


def _restore_one(item: object, kept: list[Attachment], warnings: list[str], cwd: Path) -> None:
    parsed = _parse_stored(item)
    if parsed is None:
        warnings.append("warning: dropped invalid dataset record")
        return
    resolved = _resolved_dataset(parsed, cwd)
    if resolved is None:
        warnings.append(f"warning: dropped missing dataset {_show_text(Path(parsed.path).name)}")
        return
    if not _kind_matches(resolved, parsed.kind):
        warnings.append("warning: dropped invalid dataset record")
        return
    problem = _stored_problem(resolved)
    if problem is not None:
        warnings.append(f"warning: dropped missing dataset {_show_text(resolved.name)}")
        return
    kept.append(_restore_row(_with_bound_columns(replace(parsed, path=str(resolved)))))


def _resolved_dataset(row: Attachment, cwd: Path) -> Path | None:
    kind = row.kind if row.kind in {"csv", "parquet"} else "csv"
    kept = restore_datasets([{"path": row.path, "kind": kind, "label": "file"}], cwd=cwd)
    if not kept:
        return None
    return Path(kept[0].path)


def _kind_matches(path: Path, kind: str) -> bool:
    suffix = path.suffix.lower()
    if kind == "csv":
        return suffix == ".csv"
    if kind == "parquet":
        return suffix == ".parquet"
    return False


def _restore_row(row: Attachment) -> Attachment:
    digest = _digest(Path(row.path))
    stale = row.stale or digest != row.sha256
    return replace(row, stale=stale)


def _parse_stored(item: object) -> Attachment | None:
    if not isinstance(item, dict):
        return None
    path, kind, digest = _stored_strings(item)
    if path is None or kind is None or digest is None:
        return None
    columns = _stored_columns(item.get("columns"))
    if columns is None:
        columns = ()
    return _stored_attachment(
        item,
        path,
        kind,
        digest,
        _as_int(item.get("size")),
        _optional_rows(item.get("rows")),
        columns,
    )


def _stored_strings(item: dict[object, object]) -> tuple[str | None, str | None, str | None]:
    path = item.get("path")
    kind = item.get("kind")
    digest = item.get("sha256")
    if not isinstance(path, str) or not path or len(path) > MAX_PATH_CHARS:
        return None, None, None
    if kind not in {"csv", "parquet"}:
        return None, None, None
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        return None, None, None
    return path, kind, digest.lower()


def _stored_columns(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or len(value) > MAX_COLUMNS:
        return None
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        names.append(_clean_name(item))
    return tuple(names)


def _optional_rows(value: object) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _stored_attachment(
    item: dict[object, object],
    path: str,
    kind: str,
    digest: str,
    size: int | None,
    rows: int | None,
    columns: tuple[str, ...],
) -> Attachment:
    column = item.get("available_time_col")
    issue = column if isinstance(column, str) else ""
    return Attachment(
        path=path,
        kind=kind,
        label="file",
        sha256=digest,
        size=size,
        rows=rows,
        columns=columns,
        stale=item.get("stale") is True,
        has_time=item.get("has_time") is True,
        time_monotonic=item.get("time_monotonic") is True,
        has_nulls=item.get("has_nulls") is True,
        has_available_time=item.get("has_available_time") is True,
        available_time_col=issue,
    )


def _stored_problem(path: Path) -> str | None:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.lstat()
    except OSError:
        return "file does not exist"
    if resolved != path:
        return "file does not exist"
    return _file_reason(resolved, info)


def _digest_state(row: Attachment) -> tuple[bool, bool]:
    path = Path(row.path)
    if _stored_problem(path) is not None:
        return True, True
    digest = _digest(path)
    return digest != row.sha256 or row.stale, False


def _digest(path: Path) -> str | None:
    try:
        return file_sha256(path)
    except OSError:
        return None


def _stale_line(index: int, *, missing: bool) -> str:
    if missing:
        return f"dataset is missing; /data inspect {index} then re-attach"
    return f"dataset changed since attach; /data inspect {index} then re-attach"


def _ref(row: Attachment) -> DatasetRef:
    return DatasetRef(
        path=row.path,
        kind=row.kind,
        label="file",
        available_time_col=_bound_issue_column(row),
    )


def _context_line(row: Attachment) -> str:
    columns = ", ".join(row.columns) or "-"
    return f"label=file rows={_row_count(row)} columns={columns}"


def _row_count(row: Attachment) -> str:
    if row.rows is None:
        return "unknown"
    return str(row.rows)


def _clean_name(name: str) -> str:
    text = _show_text(name)
    if not text or text == "-":
        return "?"
    return text[:MAX_COLUMN_CHARS]


def _show_text(value: str) -> str:
    cleaned = sanitize_for_terminal(redact_text(value))
    visible = "".join(ch for ch in cleaned if ch.isprintable() or ch == " ")
    return " ".join(visible.split())[:MAX_PATH_CHARS]


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    if value > _MAX_COUNTED:
        return None
    return value


def _first_line(text: str) -> str:
    line = text.splitlines()[0] if text else ""
    return line or "refusing dataset"


def _hash_fd(descriptor: int) -> str:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise OSError(errno.EINVAL, "not a regular file")
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, _READ)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _states_pit_unchecked(summary: str) -> bool:
    return "point-in-time validity is not verified" in summary.casefold()


def _same_stored_path(left: str, right: str) -> bool:
    return os.path.normcase(left) == os.path.normcase(right)


def _column_bindable(name: str, columns: tuple[str, ...]) -> str:
    if not name or name not in columns or _column_rejected(name):
        return ""
    return name


def _column_shape_rejected(name: str) -> bool:
    if not name or len(name) > MAX_COLUMN_CHARS or name != name.strip():
        return True
    return name.startswith("-") or any(char.isspace() for char in name)


def _column_rejected(name: str) -> bool:
    if _column_shape_rejected(name):
        return True
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        return True
    return _SHELL_META.search(name) is not None


def _confirmed_columns(path: Path, names: tuple[str, ...]) -> tuple[str, ...]:
    observed = _header_names(path)
    if observed is None:
        return names
    return tuple(name for name in names if name in observed)


def _header_names(path: Path) -> frozenset[str] | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_header(path)
    if suffix == ".parquet":
        return _parquet_header(path)
    return None


def _csv_header(path: Path) -> frozenset[str] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            line = handle.readline(_HEADER_CAP)
    except (OSError, UnicodeError):
        return None
    if not line:
        return frozenset()
    if len(line) >= _HEADER_CAP:
        return None
    return _header_cells(line)


def _header_cells(line: str) -> frozenset[str] | None:
    try:
        cells = next(csv.reader([line]))
    except csv.Error:
        return None
    return frozenset(_clean_name(cell) for cell in cells if cell.strip())


def _parquet_header(path: Path) -> frozenset[str] | None:
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
    except Exception:
        return None
    names = getattr(schema, "names", None)
    if not isinstance(names, list):
        return None
    return frozenset(_clean_name(name) for name in names if isinstance(name, str))


def _with_bound_columns(row: Attachment) -> Attachment:
    names = _confirmed_columns(Path(row.path), row.columns)
    column = _column_bindable(row.available_time_col, names)
    return replace(row, columns=names, available_time_col=column)


def _bound_issue_column(row: Attachment) -> str:
    names = _confirmed_columns(Path(row.path), row.columns)
    return _column_bindable(row.available_time_col, names)

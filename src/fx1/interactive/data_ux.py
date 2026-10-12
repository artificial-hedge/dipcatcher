"""Session dataset attachments for ``/data`` and ``dipcatcher exec --data``.

Profiles come from the registered ``score-forecasts --inspect`` harness
command. Harness stdout is untrusted. The sha256 recorded at attach is
rechecked before a run; a different file is not scored.
"""

from __future__ import annotations

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
_PROFILE_FAILED = "dataset profile failed"
_CHANGED_DURING = "dataset changed during profile; retry /data attach"
_MAX_PROFILE_CHARS = 1_000_000
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_READ = 1024 * 1024


@dataclass(frozen=True)
class Attachment:
    """One attached file. ``path`` is the resolved regular file."""

    path: str
    kind: str
    label: str
    sha256: str
    size: int
    rows: int
    columns: tuple[str, ...]
    stale: bool
    has_time: bool
    time_monotonic: bool
    has_nulls: bool
    has_available_time: bool

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
        }


@dataclass(frozen=True)
class _Profile:
    rows: int
    columns: tuple[str, ...]
    has_nulls: bool
    has_time: bool
    time_monotonic: bool
    has_available_time: bool
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


def restore_stored(value: object) -> tuple[list[Attachment], tuple[str, ...]]:
    """Rebuild attachments. Missing files are dropped; a new digest is stale."""
    if not isinstance(value, list):
        return [], ()
    return _restore_list(value)


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
    if not unchecked or "point-in-time" in summary.casefold():
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
    """Streaming sha256 of a regular file the caller already accepted."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_command(rest: str) -> tuple[str, str]:
    parts = shlex.split(rest)
    if len(parts) == 1 and parts[0] == "list":
        return "list", ""
    if len(parts) == 2 and parts[0] in {"attach", "remove", "inspect"}:
        return parts[0], parts[1]
    raise ValueError(_USAGE)


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
    profile, failure = _load_profile(harness, path)
    if profile is None:
        return rows, failure, False
    item = _from_profile(path, profile)
    return _upsert(rows, item), _render_profile(profile), True


def resolve_dataset(raw: str, *, cwd: Path | None = None) -> tuple[Path | None, str | None]:
    """Accept one CSV or Parquet file. The reason is a single refusal line."""
    reason = _raw_reason(raw)
    if reason is not None:
        return None, reason
    candidate = _candidate(raw, Path.cwd() if cwd is None else cwd)
    try:
        info = candidate.lstat()
    except OSError:
        return None, "file does not exist"
    early = _link_reason(candidate, info)
    if early is not None:
        return None, early
    try:
        resolved = candidate.resolve(strict=True)
        final = resolved.lstat()
    except OSError:
        return None, "file does not exist"
    problem = _file_reason(resolved, final)
    if problem is not None:
        return None, problem
    return resolved, None


def _raw_reason(raw: str) -> str | None:
    if "\x00" in raw:
        return "path contains NUL"
    if "\n" in raw or "\r" in raw:
        return "path contains a newline"
    if not raw:
        return "file does not exist"
    return None


def _candidate(raw: str, cwd: Path) -> Path:
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded
    return cwd / expanded


def _link_reason(path: Path, info: os.stat_result) -> str | None:
    """Refuse a symlink. The message names the real path and does not attach it."""
    if stat.S_ISLNK(info.st_mode):
        return _symlink_line(path)
    return _special_reason(info)


def _symlink_line(path: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return "refusing a symlink; the target does not exist"
    shown = _show_text(str(resolved))
    return f"refusing a symlink; re-attach the real file {shown}"


def _file_reason(path: Path, info: os.stat_result) -> str | None:
    special = _special_reason(info)
    if special is not None:
        return special
    if path.suffix.lower() not in {".csv", ".parquet"}:
        return "dataset must be a .csv or .parquet file"
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
    if any(row.path == str(path) for row in rows):
        return False
    return len(rows) >= MAX_DATASETS


def _load_profile(harness: Any, path: Path) -> tuple[_Profile | None, str]:
    before = _digest(path)
    if before is None:
        return None, _PROFILE_FAILED
    code, stdout = _run_inspect(harness, path)
    if _digest(path) != before:
        return None, _CHANGED_DURING
    if code != 0:
        return None, _PROFILE_FAILED
    try:
        size = path.stat().st_size
    except OSError:
        return None, _PROFILE_FAILED
    profile = parse_profile(stdout, before, size)
    if profile is None:
        return None, _PROFILE_FAILED
    return profile, ""


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
    """Defensive ``dataset_profile.v1`` parse. ``None`` means do not attach."""
    payload = _json_object(stdout)
    if payload is None or not _profile_identity(payload, digest, size):
        return None
    columns = _profile_columns(payload.get("columns"))
    timing = _time_view(payload.get("time"))
    available = _available_flag(payload.get("suggested_roles"))
    row_count = _as_int(payload.get("n_rows"))
    if columns is None or timing is None or available is None or row_count is None:
        return None
    names, nulls = columns
    has_time, monotonic = timing
    return _Profile(
        rows=row_count,
        columns=names,
        has_nulls=nulls,
        has_time=has_time,
        time_monotonic=monotonic,
        has_available_time=available,
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


def _from_profile(path: Path, profile: _Profile) -> Attachment:
    kind = "parquet" if path.suffix.lower() == ".parquet" else "csv"
    return Attachment(
        path=str(path),
        kind=kind,
        label=_label_for(path),
        sha256=profile.sha256,
        size=profile.size,
        rows=profile.rows,
        columns=profile.columns,
        stale=False,
        has_time=profile.has_time,
        time_monotonic=profile.time_monotonic,
        has_nulls=profile.has_nulls,
        has_available_time=profile.has_available_time,
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
    profile, failure = _load_profile(harness, path)
    if profile is None:
        return [failure]
    return _render_profile(profile).splitlines()


def _list_text(rows: Sequence[Attachment]) -> str:
    if not rows:
        return "no datasets"
    return "\n".join(_list_line(index, row) for index, row in enumerate(rows, start=1))


def _list_line(index: int, row: Attachment) -> str:
    digest = row.sha256[:8] if row.sha256 else "-"
    marker = " stale" if row.stale else ""
    return (
        f"{index} {row.kind} {_show_text(row.label)} rows={row.rows} "
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
        if row.path == item.path:
            updated = list(rows)
            updated[index] = item
            return updated
    return [*rows, item]


def _restore_list(value: list[object]) -> tuple[list[Attachment], tuple[str, ...]]:
    kept: list[Attachment] = []
    warnings: list[str] = []
    for item in value:
        if len(kept) >= MAX_DATASETS:
            break
        _restore_one(item, kept, warnings)
    return kept, tuple(warnings)


def _restore_one(item: object, kept: list[Attachment], warnings: list[str]) -> None:
    parsed = _parse_stored(item)
    if parsed is None:
        warnings.append("warning: dropped invalid dataset record")
        return
    problem = _stored_problem(Path(parsed.path))
    if problem is not None:
        warnings.append(f"warning: dropped missing dataset {_show_text(parsed.label)}")
        return
    kept.append(_restore_row(parsed))


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
    row_count = _as_int(item.get("rows"))
    size = _as_int(item.get("size"))
    columns = _stored_columns(item.get("columns"))
    if row_count is None or size is None or columns is None:
        return None
    return _stored_attachment(item, path, kind, digest, size, row_count, columns)


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


def _stored_attachment(
    item: dict[object, object],
    path: str,
    kind: str,
    digest: str,
    size: int,
    rows: int,
    columns: tuple[str, ...],
) -> Attachment:
    label = _clean_name(str(item.get("label") or ""))
    return Attachment(
        path=path,
        kind=kind,
        label=label if label != "?" else "dataset",
        sha256=digest,
        size=size,
        rows=rows,
        columns=columns,
        stale=item.get("stale") is True,
        has_time=item.get("has_time") is True,
        time_monotonic=item.get("time_monotonic") is True,
        has_nulls=item.get("has_nulls") is True,
        has_available_time=item.get("has_available_time") is True,
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
    return DatasetRef(path=row.path, kind=row.kind, label=row.label)


def _context_line(row: Attachment) -> str:
    columns = ", ".join(row.columns) or "-"
    return f"label={row.label} rows={row.rows} columns={columns}"


def _label_for(path: Path) -> str:
    label = _clean_name(path.stem)
    if label == "?":
        return "dataset"
    return label


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
    return value


def _first_line(text: str) -> str:
    line = text.splitlines()[0] if text else ""
    return line or "refusing dataset"

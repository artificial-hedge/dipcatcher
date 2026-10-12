"""Redacted local diagnostics report and support bundle.

The report is presence-only: versions, paths, permission findings, env var
*names*, and a harness-registry digest. Secret values are omitted. Proxy URLs
are reduced to scheme and host. The zip bundle is written atomically, mode
``0600``, and is never uploaded. ``fx1.interactive.terminal`` is intentionally
not imported; pass ``terminal_caps`` from the caller.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import io
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import fx1
from fx1.harness import HARNESS_REGISTRY, HarnessCommand
from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS
from fx1.interactive.atomicstore import FILE_MODE, ensure_private_dir, permission_problem
from fx1.interactive.redact import redact_text
from fx1.interactive.runctx import is_valid_id
from fx1.interactive.telemetry import Telemetry, format_timestamp, sanitize_value

TELEMETRY_TAIL = 100
EXECUTABLE_NAMES: tuple[str, ...] = ("dipcatcher", "quant", "fxi", "fx1")
PERMISSION_TARGETS: tuple[tuple[str, str], ...] = (
    ("credentials", "credentials.json"),
    ("sessions", "sessions"),
    ("jobs", "jobs"),
    ("logs", "logs"),
    ("history", "history"),
)
# Names only. Values are never copied, except proxy URLs reduced to scheme+host.
DIAGNOSTIC_ENV_VARS: frozenset[str] = frozenset(
    {
        "MOONSHOT_API_KEY",
        "FX1_BASE_URL",
        "FX1_CONFIG_DIR",
        "FX1_SDK_STATE_DIR",
        "FX1_CHECKPOINT_DIR",
        "FX1_BYOK_BASE_URL",
        "FX1_BYOK_API_KEY",
        "FX1_BYOK_MODEL",
        "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
        "FX1_LOCAL_SERVE_URL",
        "FX1_LOCAL_SERVE_CMD",
        "FX1_LOCAL_MODEL",
        "FX1_LOCAL_API_KEY",
        "FX1_LOCAL_TIMEOUT_S",
        "FX1_LOCAL_START_TIMEOUT_S",
        "FX1_API_KEY",
        "FX1_SIGNING_KEY",
        "NO_COLOR",
        "FORCE_COLOR",
        "COLORFGBG",
        "FXI_ORBS",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
        "no_proxy",
    }
)
DIAGNOSTIC_ENV_PREFIXES: tuple[str, ...] = ("DIPCATCHER_",)
# NO_PROXY is a host list, not a URL; presence only.
PROXY_ENV_VARS: frozenset[str] = frozenset(
    {
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    }
)
_ENDPOINT_KEYS: tuple[str, ...] = ("model", "host", "status", "fingerprint")
_SAFE_ARC = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_MIN_WIDTH = 20
_MAX_WIDTH = 500

WhichFn = Callable[[str], str | None]
Clock = Callable[[], datetime]


def harness_registry_digest() -> str:
    """SHA-256 over sorted ``(name, role, argv, timeout)`` rows from ``HARNESS_REGISTRY``."""
    rows = sorted(_registry_row(command) for command in HARNESS_REGISTRY)
    encoded = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def redact_jsonl(text: str) -> str:
    """Rewrite JSONL so each line is sanitized. Unparsed lines keep only their length."""
    kept = [_redact_jsonl_line(line) for line in text.splitlines() if line.strip()]
    if not kept:
        return ""
    return "\n".join(kept) + "\n"


def collect_diagnostics(
    *,
    config_dir: Path | None = None,
    session_id: str | None = None,
    model: str | None = None,
    endpoint_rows: Callable[[], Sequence[Mapping[str, str]]] | None = None,
    job_counts: Callable[[], Mapping[str, int]] | None = None,
    session_counts: Callable[[], Mapping[str, int]] | None = None,
    telemetry: Telemetry | None = None,
    environ: Mapping[str, str] | None = None,
    terminal_caps: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
    which: WhichFn | None = None,
) -> dict[str, Any]:
    """Presence-only diagnostics. Injected callables may be omitted.

    Every string in the result has passed through secret redaction. Invalid
    session ids are dropped rather than echoed. Callable failures are recorded
    as an error type name, never the exception message.
    """
    env = os.environ if environ is None else environ
    root = config_dir if config_dir is not None else _default_config(env)
    diag: dict[str, Any] = {
        "generated_at": _generated_at(clock),
        "fx1_version": fx1.__version__,
        "python": _python_facts(),
        "platform": _platform_facts(),
        "executables": _executables(which),
        "config_dir": str(root),
        "model": model,
        "permissions": _permissions(root),
        "endpoints": _safe_endpoints(endpoint_rows),
        "environment": _environment(env),
        "harness": _harness_facts(),
        "honesty": {"forbidden_headline_tokens": sorted(FORBIDDEN_HEADLINE_TOKENS)},
        "terminal": _terminal_block(terminal_caps),
        "jobs": _counts(job_counts),
        "sessions": _counts(session_counts),
        "telemetry": _telemetry_block(telemetry),
    }
    diag.update(_session_fields(session_id))
    safe = sanitize_value(diag)
    if isinstance(safe, dict):
        return safe
    return {}


def format_report(diag: Mapping[str, Any], *, width: int = 100) -> str:
    """Plain-text rendering of *diag*. No ANSI. Lines are wrapped to *width*."""
    safe = sanitize_value(diag)
    body: Mapping[str, Any] = safe if isinstance(safe, Mapping) else {}
    text = "\n".join(_wrap_lines(_report_lines(body), _clamp_width(width)))
    return redact_text(text + "\n")


def write_bundle(
    diag: Mapping[str, Any],
    path: Path,
    *,
    telemetry_files: Sequence[Path] = (),
) -> Path:
    """Write a mode-``0600`` zip atomically. Raises ``FileExistsError`` if *path* exists.

    Members: ``diagnostics.json``, ``report.txt``, ``telemetry/<file>`` (redacted
    line by line), and ``MANIFEST.json`` with sha256 of every other member.
    """
    if os.path.lexists(path):
        raise FileExistsError(path)
    _publish_private(path, _zip_bytes(diag, telemetry_files))
    return path


def _registry_row(command: HarnessCommand) -> tuple[str, str, tuple[str, ...], int]:
    argv = tuple(str(part) for part in command.argv)
    return (command.name, str(command.role), argv, int(command.timeout_s))


def _redact_jsonl_line(raw: str) -> str:
    try:
        return _render_jsonl(raw)
    except (TypeError, ValueError):
        return _json_text({"event": "telemetry.unparsed", "line_len": len(raw)})


def _render_jsonl(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _json_text({"event": "telemetry.unparsed", "line_len": len(raw)})
    return _json_text(sanitize_value(parsed))


def _json_text(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_type_name,
    )


def _json_bytes(obj: Any) -> bytes:
    text = json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=_type_name,
    )
    return (text + "\n").encode("utf-8")


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def _default_config(env: Mapping[str, str]) -> Path:
    override = env.get("FX1_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fx1"


def _generated_at(clock: Clock | None) -> str:
    moment = datetime.now(UTC)
    if clock is None:
        return format_timestamp(moment)
    try:
        candidate = clock()
    except Exception:
        return format_timestamp(moment)
    if isinstance(candidate, datetime):
        return format_timestamp(candidate)
    return format_timestamp(moment)


def _python_facts() -> dict[str, str]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _platform_facts() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "summary": platform.platform(),
    }


def _executables(which: WhichFn | None) -> dict[str, str | None]:
    lookup: WhichFn = shutil.which if which is None else which
    return {name: _lookup_one(lookup, name) for name in EXECUTABLE_NAMES}


def _lookup_one(lookup: WhichFn, name: str) -> str | None:
    try:
        found = lookup(name)
    except Exception:
        return None
    if isinstance(found, str) and found:
        return found
    return None


def _permissions(root: Path) -> list[dict[str, Any]]:
    """List records. Slot names stay values: the key ``credentials`` is secret-shaped."""
    return [_permission_slot(slot, root / rel) for slot, rel in PERMISSION_TARGETS]


def _permission_slot(slot: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    problem = permission_problem(path) if exists else None
    return {"slot": slot, "path": path.name, "exists": exists, "problem": problem}


def _session_fields(session_id: str | None) -> dict[str, Any]:
    if session_id is None:
        return {"session_id": None, "session_id_rejected": False}
    if is_valid_id(session_id, "s"):
        return {"session_id": session_id, "session_id_rejected": False}
    return {"session_id": None, "session_id_rejected": True}


def _safe_endpoints(
    source: Callable[[], Sequence[Mapping[str, str]]] | None,
) -> list[dict[str, str]]:
    if source is None:
        return []
    try:
        return _endpoint_rows(source())
    except Exception as exc:
        return [{"status": "unavailable", "error_type": type(exc).__name__}]


def _endpoint_rows(rows: Any) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item = _endpoint_row(row)
        if item:
            picked.append(item)
    return picked


def _endpoint_row(row: Mapping[str, Any]) -> dict[str, str]:
    picked: dict[str, str] = {}
    for key in _ENDPOINT_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            picked[key] = value
    return picked


def _environment(environ: Mapping[str, str]) -> list[dict[str, Any]]:
    names = [name for name in environ if isinstance(name, str) and _env_selected(name)]
    return [_env_row(name, _as_text(environ[name])) for name in sorted(names)]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _env_selected(name: str) -> bool:
    return name in DIAGNOSTIC_ENV_VARS or name.startswith(DIAGNOSTIC_ENV_PREFIXES)


def _env_row(name: str, value: str) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "set": True}
    if name not in PROXY_ENV_VARS:
        return row
    endpoint = _proxy_endpoint(value)
    if endpoint is not None:
        row["endpoint"] = endpoint
    return row


def _proxy_endpoint(value: str) -> str | None:
    parsed = _parse_proxy(value)
    if parsed is None:
        return None
    scheme, host, port = parsed
    endpoint = _format_endpoint(scheme, host, port)
    if redact_text(endpoint) != endpoint:
        return None
    return endpoint


def _parse_proxy(value: str) -> tuple[str, str, int | None] | None:
    raw = value.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlsplit(raw)
    host = parsed.hostname
    if not host:
        return None
    return parsed.scheme or "http", host, _port(parsed)


def _port(parsed: Any) -> int | None:
    try:
        port = parsed.port
    except ValueError:
        return None
    if isinstance(port, int):
        return port
    return None


def _format_endpoint(scheme: str, host: str, port: int | None) -> str:
    token = f"[{host}]" if ":" in host else host
    if port is None:
        return f"{scheme}://{token}"
    return f"{scheme}://{token}:{port}"


def _counts(source: Callable[[], Mapping[str, int]] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    try:
        raw = source()
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__}
    if not isinstance(raw, Mapping):
        return {"ok": False, "error_type": "TypeError"}
    return _int_map(raw)


def _int_map(raw: Mapping[Any, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    return out


def _terminal_block(caps: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if caps is None:
        return None
    return dict(caps)


def _harness_facts() -> dict[str, Any]:
    return {"command_count": len(HARNESS_REGISTRY), "registry_sha256": harness_registry_digest()}


def _telemetry_block(telemetry: Telemetry | None) -> dict[str, Any]:
    if telemetry is None:
        return {"events": [], "dropped": 0, "sink_error": None, "level": None}
    return {
        "events": _telemetry_events(telemetry),
        "dropped": telemetry.dropped,
        "sink_error": telemetry.sink_error,
        "level": telemetry.level.name.lower(),
    }


def _telemetry_events(telemetry: Telemetry) -> list[dict[str, Any]]:
    try:
        rows = telemetry.tail(TELEMETRY_TAIL)
    except Exception:
        return []
    return [row for row in rows if isinstance(row, dict)]


def _clamp_width(width: int) -> int:
    if width < _MIN_WIDTH:
        return _MIN_WIDTH
    if width > _MAX_WIDTH:
        return _MAX_WIDTH
    return width


def _report_lines(diag: Mapping[str, Any]) -> list[str]:
    lines = ["dipcatcher diagnostics", ""]
    lines.extend(_identity_lines(diag))
    lines.extend(_executable_lines(diag))
    lines.extend(_permission_lines(diag))
    lines.extend(_endpoint_lines(diag))
    lines.extend(_environment_lines(diag))
    lines.extend(_harness_lines(diag))
    lines.extend(_honesty_lines(diag))
    lines.extend(_terminal_lines(diag))
    lines.extend(_count_lines("jobs", diag.get("jobs")))
    lines.extend(_count_lines("sessions", diag.get("sessions")))
    lines.extend(_telemetry_lines(diag.get("telemetry")))
    return lines


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _identity_lines(diag: Mapping[str, Any]) -> list[str]:
    python = _mapping(diag.get("python"))
    plat = _mapping(diag.get("platform"))
    lines = [
        f"generated_at: {_fmt(diag.get('generated_at'))}",
        f"fx1_version: {_fmt(diag.get('fx1_version'))}",
        f"python: {_fmt(python.get('implementation'))} {_fmt(python.get('version'))}",
        f"executable: {_fmt(python.get('executable'))}",
        (
            f"platform: {_fmt(plat.get('system'))} {_fmt(plat.get('release'))} "
            f"{_fmt(plat.get('machine'))}"
        ),
        f"config_dir: {_fmt(diag.get('config_dir'))}",
        f"session_id: {_fmt(diag.get('session_id'))}",
        f"model: {_fmt(diag.get('model'))}",
    ]
    if diag.get("session_id_rejected") is True:
        lines.append("session_id_rejected: true")
    lines.append("")
    return lines


def _executable_lines(diag: Mapping[str, Any]) -> list[str]:
    found = diag.get("executables")
    lines = ["executables:"]
    if not isinstance(found, Mapping):
        return [*lines, "  (unavailable)", ""]
    for name in EXECUTABLE_NAMES:
        path = found.get(name)
        shown = path if isinstance(path, str) else "(not found)"
        lines.append(f"  {name}: {shown}")
    lines.append("")
    return lines


def _permission_lines(diag: Mapping[str, Any]) -> list[str]:
    found = diag.get("permissions")
    lines = ["permissions:"]
    if not isinstance(found, list):
        return [*lines, "  (unavailable)", ""]
    by_slot = _permission_index(found)
    for slot, _rel in PERMISSION_TARGETS:
        lines.append(f"  {slot}: {_permission_text(by_slot.get(slot, {}))}")
    lines.append("")
    return lines


def _permission_index(found: list[Any]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in found:
        if not isinstance(item, Mapping):
            continue
        slot = item.get("slot")
        if isinstance(slot, str):
            indexed[slot] = item
    return indexed


def _permission_text(slot: Mapping[str, Any]) -> str:
    if slot.get("exists") is not True:
        return "absent"
    problem = slot.get("problem")
    if isinstance(problem, str) and problem:
        return problem
    return "ok"


def _endpoint_lines(diag: Mapping[str, Any]) -> list[str]:
    rows = diag.get("endpoints")
    lines = ["endpoints:"]
    if not isinstance(rows, list) or not rows:
        return [*lines, "  (none)", ""]
    for row in rows:
        lines.append(f"  {_fmt(row)}")
    lines.append("")
    return lines


def _environment_lines(diag: Mapping[str, Any]) -> list[str]:
    rows = diag.get("environment")
    lines = ["environment (presence only):"]
    if not isinstance(rows, list) or not rows:
        return [*lines, "  (none set)", ""]
    for row in rows:
        if isinstance(row, Mapping):
            lines.append(_env_line(row))
    lines.append("")
    return lines


def _env_line(row: Mapping[str, Any]) -> str:
    name = row.get("name")
    endpoint = row.get("endpoint")
    if isinstance(endpoint, str):
        return f"  {name} endpoint={endpoint}"
    return f"  {name}"


def _harness_lines(diag: Mapping[str, Any]) -> list[str]:
    harness = diag.get("harness")
    if not isinstance(harness, Mapping):
        return ["harness: (unavailable)", ""]
    return [
        "harness:",
        f"  commands: {_fmt(harness.get('command_count'))}",
        f"  registry_sha256: {_fmt(harness.get('registry_sha256'))}",
        "",
    ]


def _honesty_lines(diag: Mapping[str, Any]) -> list[str]:
    honesty = _mapping(diag.get("honesty"))
    tokens = honesty.get("forbidden_headline_tokens")
    shown = (
        ", ".join(str(token) for token in tokens) if isinstance(tokens, list) else "(unavailable)"
    )
    return ["honesty:", f"  forbidden_headline_tokens: {shown}", ""]


def _terminal_lines(diag: Mapping[str, Any]) -> list[str]:
    caps = diag.get("terminal")
    if not isinstance(caps, Mapping) or not caps:
        return ["terminal: (none)", ""]
    return ["terminal:", f"  {_fmt(dict(caps))}", ""]


def _count_lines(title: str, value: Any) -> list[str]:
    lines = [f"{title}:"]
    if not isinstance(value, Mapping):
        return [*lines, "  (unavailable)", ""]
    if not value:
        return [*lines, "  (none)", ""]
    for key in sorted(str(item) for item in value):
        lines.append(f"  {key}: {_fmt(value.get(key))}")
    lines.append("")
    return lines


def _telemetry_lines(block: Any) -> list[str]:
    lines = ["telemetry:"]
    if not isinstance(block, Mapping):
        return [*lines, "  (none)", ""]
    lines.append(f"  dropped: {_fmt(block.get('dropped'))}")
    lines.append(f"  level: {_fmt(block.get('level'))}")
    sink = block.get("sink_error")
    if isinstance(sink, Mapping):
        lines.append(f"  sink_error: {_fmt(dict(sink))}")
    events = block.get("events")
    if isinstance(events, list):
        lines.append(f"  events: {len(events)}")
        for event in events:
            lines.append(f"  {_fmt(event)}")
    lines.append("")
    return lines


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return _fmt_scalar(value)
    return _fmt_json(value)


def _fmt_scalar(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fmt_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_type_name)
    except (TypeError, ValueError):
        return _type_name(value)


def _wrap_lines(lines: Sequence[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if line == "":
            wrapped.append("")
            continue
        parts = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        wrapped.extend(parts or [""])
    return wrapped


def _zip_bytes(diag: Mapping[str, Any], telemetry_files: Sequence[Path]) -> bytes:
    safe = sanitize_value(diag)
    body: Mapping[str, Any] = safe if isinstance(safe, Mapping) else {}
    members = _bundle_members(body, telemetry_files)
    members.append(("MANIFEST.json", _manifest_bytes(members)))
    return _pack(members)


def _bundle_members(
    diag: Mapping[str, Any], telemetry_files: Sequence[Path]
) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = [
        ("diagnostics.json", _json_bytes(diag)),
        ("report.txt", format_report(diag).encode("utf-8")),
    ]
    used: set[str] = set()
    for src in telemetry_files:
        members.append((_arc_name(Path(src), used), _telemetry_member(Path(src))))
    return members


def _manifest_bytes(members: Sequence[tuple[str, bytes]]) -> bytes:
    files = [
        {"bytes": len(data), "name": name, "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in members
    ]
    return _json_bytes({"files": files})


def _telemetry_member(path: Path) -> bytes:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _json_bytes({"event": "telemetry.unreadable", "reason": type(exc).__name__})
    return redact_jsonl(text).encode("utf-8")


def _arc_name(path: Path, used: set[str]) -> str:
    base = _safe_base(path.name)
    candidate = f"telemetry/{base}"
    stem = _safe_base(path.stem)
    suffix = path.suffix if _SAFE_ARC.fullmatch(path.suffix) else ".jsonl"
    index = 2
    while candidate in used:
        candidate = f"telemetry/{stem}-{index}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _safe_base(name: str) -> str:
    if _SAFE_ARC.fullmatch(name) and name not in {".", ".."} and ".." not in name:
        return name
    return "events.jsonl"


def _pack(members: Sequence[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            info = zipfile.ZipInfo(filename=name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def _publish_private(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    ensure_private_dir(path.parent)
    tmp = _write_temp(path.parent, data)
    try:
        _link_new(tmp, path)
    finally:
        _discard(tmp)


def _write_temp(directory: Path, data: bytes) -> Path:
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".dipcatcher-diag.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        _write_fd(fd, data)
        _chmod_private(tmp)
    except Exception:
        _discard(tmp)
        raise
    return tmp


def _write_fd(fd: int, data: bytes) -> None:
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _chmod_private(path: Path) -> None:
    if os.name == "posix":
        os.chmod(path, FILE_MODE)


def _link_new(tmp: Path, path: Path) -> None:
    try:
        os.link(tmp, path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise FileExistsError(path) from exc
        raise


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


__all__ = [
    "DIAGNOSTIC_ENV_PREFIXES",
    "DIAGNOSTIC_ENV_VARS",
    "EXECUTABLE_NAMES",
    "PERMISSION_TARGETS",
    "PROXY_ENV_VARS",
    "TELEMETRY_TAIL",
    "collect_diagnostics",
    "format_report",
    "harness_registry_digest",
    "redact_jsonl",
    "write_bundle",
]

#!/usr/bin/env python3
"""Back up the lab data directory, destroy it, bring prod up, and restore.

Stdout is one JSON object. ``rto_seconds`` is the wall time from the start
of the destroy through a matching ledger digest and both TLS health checks.
The earlier stop, taken so the writer is not replacing a file during the
backup, is not included. The process exits 0 only when that restore matches
and ``prod-down`` exits 0. API key values are not part of the JSON.

The data paths are the bind-mounted lab and research directories. They are
removed only when the resolved path is under ``/tmp`` and a path component
starts with ``dipcatcher-``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

_ROOT = Path(__file__).resolve().parents[1]
_SEED_RELATIVE = "metadata/paper/dr-drill/broker_state.json"
_SEED_BODY = '{"data_label":"SYNTHETIC","schema":"dr-drill-ledger","step":0}\n'
_SCHEMA = "paper_state_backup.v1"
_LEDGER = "broker_state.json"
_HSTS = "max-age=31536000"
_UP_TIMEOUT_S = 900
_DOWN_TIMEOUT_S = 180
_BACKUP_TIMEOUT_S = 180
_DATA_TIMEOUT_S = 120
_READY_TIMEOUT_S = 60
_KEY_ERROR = "QUANT_API_KEY and RESEARCH_API_KEY are required"
_REQUIRED = ("backup", "destroy", "up", "restore", "ledger", "health", "down")
_RTO_INCLUDES = "destroy, prod-up, restore, ledger digest, TLS health"
_SEED_CODE = (
    "import os\n"
    "from pathlib import Path\n"
    "path = Path(os.environ['DIPCATCHER_SEED_PATH'])\n"
    "path.parent.mkdir(parents=True, exist_ok=True)\n"
    "path.write_text(os.environ['DIPCATCHER_SEED_BODY'], encoding='utf-8')\n"
    "os.chmod(path, 0o644)\n"
    "os.chown(path, 10001, 10001)\n"
)


def elapsed_seconds(started: float, finished: float) -> float:
    """Seconds between two monotonic marks, rounded to milliseconds."""
    return round(finished - started, 3)


def removable(path: Path) -> bool:
    """True for a dipcatcher path under /tmp, and never for /tmp itself."""
    resolved = path.resolve()
    root = Path("/tmp").resolve()
    if resolved == root or not resolved.is_relative_to(root):
        return False
    return any(part.startswith("dipcatcher-") for part in resolved.parts)


def combine_ledger_digest(rows: list[tuple[str, str]]) -> str:
    """The ledger digest. One broker state uses its file sha256."""
    ordered = sorted((path, digest) for path, digest in rows if _is_ledger(path))
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0][1]
    blob = "\n".join(f"{path} {digest}" for path, digest in ordered).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def restored_digest(dest: Path, entries: list[tuple[str, str, int]]) -> str:
    """Re-hash the restored archive. An empty string means a mismatch."""
    rows: list[tuple[str, str]] = []
    for relative, digest, size in entries:
        target = dest.joinpath(*PurePosixPath(relative).parts)
        try:
            found, found_size = _hash_file(target)
        except OSError:
            return ""
        if found != digest or found_size != size:
            return ""
        if _is_ledger(relative):
            rows.append((relative, found))
    return combine_ledger_digest(rows)


def health_ok(status: int, headers: Mapping[str, str]) -> bool:
    """True when liveness is 200 and HSTS carries the one-year max-age."""
    folded = {key.lower(): value for key, value in headers.items()}
    return status == 200 and _HSTS in folded.get("strict-transport-security", "")


def _is_ledger(relative: str) -> bool:
    return PurePosixPath(relative).name == _LEDGER


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def _keys_present() -> bool:
    return bool(os.environ.get("QUANT_API_KEY", "").strip()) and bool(
        os.environ.get("RESEARCH_API_KEY", "").strip()
    )


def _redact(text: str) -> str:
    redacted = text
    for name in ("QUANT_API_KEY", "RESEARCH_API_KEY", "DIPCATCHER_ALERT_WEBHOOK_SECRET"):
        value = os.environ.get(name, "")
        if value:
            redacted = redacted.replace(value, f"<{name}>")
    return redacted


def _sudo_argv() -> list[str]:
    raw = os.environ.get("SUDO")
    if raw is None:
        raw = "sudo"
    raw = raw.strip()
    if not raw:
        return []
    return shlex.split(raw)


def _privileged(*args: str) -> list[str]:
    prefix = _sudo_argv()
    if not prefix:
        return list(args)
    return [*prefix, *args]


def _run(argv: list[str], timeout: float = 60, env: dict[str, str] | None = None) -> int:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=_ROOT,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1
    sys.stderr.write(_redact(proc.stdout or ""))
    sys.stderr.write(_redact(proc.stderr or ""))
    return proc.returncode


def _make(target: str, timeout: float) -> int:
    env = os.environ.copy()
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(name, None)
    return _run(["make", "-f", "deploy/prod.mk", target], timeout=timeout, env=env)


def _lab() -> Path:
    return Path(os.environ.get("DIPCATCHER_LAB_DATA", "/tmp/dipcatcher-ops7-data/lab"))


def _research() -> Path:
    return Path(os.environ.get("DIPCATCHER_RESEARCH_DATA", "/tmp/dipcatcher-ops7-data/research"))


def _backup_dir() -> Path:
    return Path(os.environ.get("BACKUP_DIR", "/tmp/dipcatcher-ops7-backup"))


def _check(name: str, ok: bool, detail: str = "") -> dict[str, object]:
    item: dict[str, object] = {"name": name, "ok": ok}
    if detail:
        item["detail"] = detail
    return item


def ledger_paths(root: Path) -> list[Path]:
    """Regular ``broker_state.json`` files under ``root``. Dot directories are skipped."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for name in filenames:
            if name == _LEDGER:
                found.append(Path(dirpath) / name)
    found.sort()
    return found


def manifest_rows(path: Path) -> list[tuple[str, str, int]]:
    """Archive entries, or an empty list when the manifest is not the backup schema."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        return []
    files = payload.get("files")
    if not isinstance(files, list):
        return []
    rows: list[tuple[str, str, int]] = []
    for entry in files:
        if not isinstance(entry, dict):
            return []
        relative = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(relative, str) or not isinstance(digest, str):
            return []
        if isinstance(size, bool) or not isinstance(size, int):
            return []
        rows.append((relative, digest, size))
    return rows


def _mode_tree(path: Path) -> bool:
    if not removable(path) or not path.exists():
        return False
    return _run(_privileged("chmod", "-R", "a+rX", str(path))) == 0


def _write_seed(lab: Path) -> bool:
    env = os.environ.copy()
    env["DIPCATCHER_SEED_PATH"] = str(lab / _SEED_RELATIVE)
    env["DIPCATCHER_SEED_BODY"] = _SEED_BODY
    prefix = _sudo_argv()
    if prefix:
        argv = [
            *prefix,
            "--preserve-env=DIPCATCHER_SEED_PATH,DIPCATCHER_SEED_BODY",
            "python3",
            "-c",
            _SEED_CODE,
        ]
    else:
        argv = ["python3", "-c", _SEED_CODE]
    return _run(argv, env=env) == 0


def _ensure_ledger(lab: Path) -> bool:
    if not removable(lab):
        return False
    if not ledger_paths(lab) and not _write_seed(lab):
        return False
    return _mode_tree(lab) and bool(ledger_paths(lab))


def _clear_backup() -> bool:
    path = _backup_dir()
    if not removable(path):
        return False
    if not path.exists() and not path.is_symlink():
        return True
    if _run(_privileged("rm", "-rf", str(path))) != 0:
        return False
    return not path.exists()


def _destroy() -> bool:
    paths = [_lab(), _research()]
    if any(not removable(path) for path in paths):
        return False
    argv = _privileged("rm", "-rf", *(str(path) for path in paths))
    if _run(argv) != 0:
        return False
    return all(not path.exists() for path in paths)


def _relax(path: Path) -> bool:
    if not removable(path) or not path.is_dir():
        return False
    chown = _run(_privileged("chown", "-R", "10001:10001", str(path)))
    return chown == 0 and _mode_tree(path)


def _request(url: str, ca_file: Path) -> tuple[int, dict[str, str]]:
    context = ssl.create_default_context(cafile=str(ca_file))
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, context=context, timeout=10) as response:
            response.read()
            return response.status, {key: value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, {key: value for key, value in exc.headers.items()}


def _request_retry(url: str, ca_file: Path) -> tuple[int, dict[str, str]]:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            return _request(url, ca_file)
        except Exception:
            if time.monotonic() >= deadline:
                return 0, {}
            time.sleep(1)


def _listeners_ok() -> bool:
    tls = Path(os.environ.get("DIPCATCHER_TLS_DIR", "/tmp/dipcatcher-ops7-tls"))
    ca_file = tls / "dev.crt"
    if not ca_file.is_file():
        return False
    lab = os.environ.get("LAB_TLS_PORT", "8443").strip() or "8443"
    research = os.environ.get("RESEARCH_TLS_PORT", "8444").strip() or "8444"
    for port in (lab, research):
        status, headers = _request_retry(f"https://127.0.0.1:{port}/health", ca_file)
        if not health_ok(status, headers):
            return False
    return True


def _prepare() -> tuple[bool, str, list[tuple[str, str, int]], dict[str, object]]:
    if _make("prod-data", _DATA_TIMEOUT_S) != 0:
        return False, "", [], _check("backup", False, "data dir")
    if _make("prod-down", _DOWN_TIMEOUT_S) != 0:
        return False, "", [], _check("backup", False, "quiesce")
    if not _ensure_ledger(_lab()):
        return False, "", [], _check("backup", False, "ledger")
    if not _clear_backup():
        return False, "", [], _check("backup", False, "archive")
    if _make("prod-backup", _BACKUP_TIMEOUT_S) != 0:
        return False, "", [], _check("backup", False, "backup")
    entries = manifest_rows(_backup_dir() / "manifest.json")
    digest = combine_ledger_digest([(path, item) for path, item, _size in entries])
    if not digest or not entries:
        return False, "", [], _check("backup", False, "digest")
    return True, digest, entries, _check("backup", True)


def _recover(
    digest: str, entries: list[tuple[str, str, int]]
) -> tuple[list[dict[str, object]], float | None, dict[str, float]]:
    checks: list[dict[str, object]] = []
    phases: dict[str, float] = {}
    started = time.monotonic()
    mark = started
    destroyed = _destroy()
    now = time.monotonic()
    phases["destroy"] = elapsed_seconds(mark, now)
    checks.append(_check("destroy", destroyed))
    if not destroyed:
        return checks, elapsed_seconds(started, now), phases
    mark = now
    up = _make("prod-up", _UP_TIMEOUT_S)
    now = time.monotonic()
    phases["up"] = elapsed_seconds(mark, now)
    checks.append(_check("up", up == 0))
    restored = _make("prod-restore", _BACKUP_TIMEOUT_S) if up == 0 else 1
    mark = now
    now = time.monotonic()
    phases["restore"] = elapsed_seconds(mark, now)
    checks.append(_check("restore", restored == 0))
    matched = ""
    if restored == 0 and _relax(_lab()):
        matched = restored_digest(_lab(), entries)
    ledger_ok = bool(digest) and matched == digest
    checks.append(_check("ledger", ledger_ok))
    mark = now
    healthy = _listeners_ok() if ledger_ok else False
    now = time.monotonic()
    phases["health"] = elapsed_seconds(mark, now)
    checks.append(_check("health", healthy))
    return checks, elapsed_seconds(started, now), phases


def _report(
    checks: list[dict[str, object]],
    digest: str,
    rto: float | None,
    phases: dict[str, float],
) -> dict[str, object]:
    by_name = {str(item["name"]): item for item in checks}
    complete = all(name in by_name and bool(by_name[name].get("ok")) for name in _REQUIRED)
    clean = bool(checks) and all(bool(item.get("ok")) for item in checks)
    return {
        "checks": checks,
        "ledger_digest": digest,
        "ok": complete and clean and rto is not None,
        "phases": phases,
        "rto_includes": _RTO_INCLUDES,
        "rto_seconds": rto,
    }


def _drill() -> dict[str, object]:
    checks: list[dict[str, object]] = []
    digest = ""
    rto: float | None = None
    phases: dict[str, float] = {}
    try:
        ok, digest, entries, backup = _prepare()
        checks.append(backup)
        if ok:
            extra, rto, phases = _recover(digest, entries)
            checks.extend(extra)
    finally:
        down = _make("prod-down", _DOWN_TIMEOUT_S)
        checks.append(_check("down", down == 0))
    return _report(checks, digest, rto, phases)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        _emit({"error": "prod-dr-drill takes no arguments", "ok": False})
        return 1
    if not _keys_present():
        _emit({"error": _KEY_ERROR, "ok": False})
        return 1
    report = _drill()
    _emit(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

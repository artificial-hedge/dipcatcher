#!/usr/bin/env python3
"""Bring the production compose stack up, check it, and bring it down.

Stdout is one JSON object. The process exits 0 only when ``prod-up`` exited
0, every check passed, and ``prod-down`` exited 0. It exits 1 otherwise.
Make and docker output goes to stderr. API key values are redacted there
and are not part of the JSON. A missing key prints the JSON error and does
not invoke make. ``prod-down`` still runs after a ``prod-up`` attempt.
"""

from __future__ import annotations

import errno
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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BODY_LIMIT = 64 * 1024
_UP_TIMEOUT_S = 900
_DOWN_TIMEOUT_S = 180
_READY_TIMEOUT_S = 30
_KEY_ERROR = "QUANT_API_KEY and RESEARCH_API_KEY are required"
_HSTS = "max-age=31536000"

_CLAIM = (
    "from quant_fund.utils.write_owner import WriteOwnerError, claim_write_lock\n"
    "try:\n"
    "    claim_write_lock()\n"
    "    print('claimed')\n"
    "except WriteOwnerError as exc:\n"
    "    print(exc)\n"
)
_READER = (
    "from pathlib import Path\n"
    "from quant_fund.utils.operator_kill import OperatorKillError, write_operator_kill\n"
    "try:\n"
    "    write_operator_kill(Path('/tmp/not-the-kill-file'), 'HALT_NEW_ORDERS')\n"
    "    print('wrote')\n"
    "except OperatorKillError as exc:\n"
    "    print(exc)\n"
)
_RESEARCH_PROBE = (
    "import os\n"
    "path = '/app/data/.dipcatcher-smoke-probe-%d' % os.getpid()\n"
    "flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY\n"
    "try:\n"
    "    fd = os.open(path, flags, 0o644)\n"
    "except OSError as exc:\n"
    "    print(0 if exc.errno is None else exc.errno)\n"
    "else:\n"
    "    os.close(fd)\n"
    "    os.unlink(path)\n"
    "    print('writable')\n"
)


def health_ok(status: int, headers: Mapping[str, str]) -> bool:
    """True when liveness is 200 and HSTS carries the one-year max-age."""
    folded = {key.lower(): value for key, value in headers.items()}
    return status == 200 and _HSTS in folded.get("strict-transport-security", "")


def body_ok(small_status: int, large_status: int) -> bool:
    """True when a small POST is not rejected and one extra byte is 413."""
    return small_status != 413 and 0 < small_status < 500 and large_status == 413


def research_mount_ok(text: str) -> bool:
    """True only for a read-only filesystem, not a mode bit the root user ignores."""
    return text.strip() == str(errno.EROFS)


def summarize(up: int, down: int, checks: list[dict[str, object]]) -> dict[str, object]:
    """One JSON-ready report. An empty check list cannot pass."""
    ok = up == 0 and down == 0 and bool(checks) and all(bool(item.get("ok")) for item in checks)
    return {"checks": checks, "down": down, "ok": ok, "up": up}


def docker_argv() -> list[str]:
    """The docker command prefix. ``DIPCATCHER_DOCKER`` wins, then ``SUDO``."""
    raw = os.environ.get("DIPCATCHER_DOCKER", "").strip()
    if raw:
        return shlex.split(raw)
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _keys_present() -> bool:
    return bool(os.environ.get("QUANT_API_KEY", "").strip()) and bool(
        os.environ.get("RESEARCH_API_KEY", "").strip()
    )


def _redact(text: str) -> str:
    redacted = text
    for name in ("QUANT_API_KEY", "RESEARCH_API_KEY"):
        value = os.environ.get(name, "")
        if value:
            redacted = redacted.replace(value, f"<{name}>")
    return redacted


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def _make(target: str, timeout: float) -> int:
    env = os.environ.copy()
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(name, None)
    try:
        proc = subprocess.run(
            ["make", "-f", "deploy/prod.mk", target],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(_redact(f"{target} timed out after {timeout:.0f}s\n"))
        return 1
    sys.stderr.write(_redact(proc.stdout or ""))
    sys.stderr.write(_redact(proc.stderr or ""))
    return proc.returncode


def _request(url: str, ca_file: Path, body: bytes | None) -> tuple[int, dict[str, str]]:
    context = ssl.create_default_context(cafile=str(ca_file))
    method = "GET" if body is None else "POST"
    request = urllib.request.Request(url, data=body, method=method)
    try:
        with urllib.request.urlopen(request, context=context, timeout=10) as response:
            response.read()
            return response.status, {key: value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, {key: value for key, value in exc.headers.items()}


def _request_retry(url: str, ca_file: Path, body: bytes | None) -> tuple[int, dict[str, str]]:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            return _request(url, ca_file, body)
        except Exception:
            if time.monotonic() >= deadline:
                return 0, {}
            time.sleep(1)


def _exec(container: str, code: str) -> str:
    cmd = [
        *docker_argv(),
        "exec",
        container,
        "/app/.venv/bin/python",
        "-c",
        code,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return ""
    return (proc.stdout or "").strip()


def _exec_until(container: str, code: str, predicate) -> str:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    last = ""
    while True:
        last = _exec(container, code)
        if predicate(last):
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(1)


def _detail(text: str) -> str:
    return _redact(text).strip().replace("\n", " ")[:180]


def _container(env_name: str, suffix: str) -> str:
    project = os.environ.get("PROJECT", "dipcatcher-ops7").strip() or "dipcatcher-ops7"
    return os.environ.get(env_name, "").strip() or f"{project}-{suffix}"


def _tls_checks(label: str, port: str, ca_file: Path) -> list[dict[str, object]]:
    base = f"https://127.0.0.1:{port}/health"
    status, headers = _request_retry(base, ca_file, None)
    small, _small_headers = _request_retry(base, ca_file, b"x" * 100)
    large, _large_headers = _request_retry(base, ca_file, b"x" * (_BODY_LIMIT + 1))
    return [
        {"name": f"{label}-health", "ok": health_ok(status, headers), "status": status},
        {
            "name": f"{label}-body",
            "ok": body_ok(small, large),
            "small": small,
            "large": large,
        },
    ]


def _token_check(name: str, container: str, code: str, predicate) -> dict[str, object]:
    text = _exec_until(container, code, predicate)
    return {"detail": _detail(text), "name": name, "ok": bool(predicate(text))}


def collect_checks() -> list[dict[str, object]]:
    """TLS health, the body cap, the lab lock, and the research read-only mount."""
    tls = os.environ.get("DIPCATCHER_TLS_DIR", "").strip()
    ca_file = Path(tls) / "dev.crt" if tls else Path()
    lab_port = os.environ.get("LAB_TLS_PORT", "8443").strip() or "8443"
    research_port = os.environ.get("RESEARCH_TLS_PORT", "8444").strip() or "8444"
    checks: list[dict[str, object]] = []
    if not tls or not ca_file.is_file():
        checks.append(
            {"name": "tls-ca", "ok": False, "detail": "DIPCATCHER_TLS_DIR/dev.crt missing"}
        )
        return checks
    checks.extend(_tls_checks("lab", lab_port, ca_file))
    checks.extend(_tls_checks("research", research_port, ca_file))
    checks.append(
        _token_check(
            "lab-a-lock",
            _container("LAB_A", "lab-a-1"),
            _CLAIM,
            lambda text: "write_owner_busy" in text,
        )
    )
    checks.append(
        _token_check(
            "lab-b-reader",
            _container("LAB_B", "lab-b-1"),
            _READER,
            lambda text: "is not the writer" in text,
        )
    )
    checks.append(
        _token_check(
            "research-ro",
            _container("RESEARCH_A", "research-a-1"),
            _RESEARCH_PROBE,
            research_mount_ok,
        )
    )
    return checks


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        _emit({"error": "prod-smoke takes no arguments", "ok": False})
        return 1
    if not _keys_present():
        _emit({"error": _KEY_ERROR, "ok": False})
        return 1
    up = 1
    down = 1
    checks: list[dict[str, object]] = []
    try:
        up = _make("prod-up", _UP_TIMEOUT_S)
        if up == 0:
            checks = collect_checks()
    finally:
        down = _make("prod-down", _DOWN_TIMEOUT_S)
    report = summarize(up, down, checks)
    _emit(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/usr/bin/env python3
"""Health gate after ``prod-rollback`` recreates the previous image.

Stdout is one JSON object. The process exits 0 only when both TLS listeners
return 200 with HSTS and the four API replicas are healthy. Make tags the
previous image and recreates the containers before it calls this script.
"""

from __future__ import annotations

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

_HSTS = "max-age=31536000"
_TIMEOUT_S = 180.0
_NAMES = ("LAB_A", "LAB_B", "RESEARCH_A", "RESEARCH_B")
_DEFAULTS = ("lab-a-1", "lab-b-1", "research-a-1", "research-b-1")


def listener_ok(status: int, headers: Mapping[str, str]) -> bool:
    """True when the proxy answers 200 and sends the one-year HSTS policy."""
    folded = {key.lower(): value for key, value in headers.items()}
    return status == 200 and _HSTS in folded.get("strict-transport-security", "")


def gate_ready(states: list[str], lab_ok: bool, research_ok: bool) -> bool:
    """True when every replica is healthy and both listeners passed."""
    if len(states) != 4:
        return False
    if any(state != "healthy" for state in states):
        return False
    return lab_ok and research_ok


def docker_argv() -> list[str]:
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


def _containers() -> list[str]:
    project = os.environ.get("PROJECT", "dipcatcher-ops7").strip() or "dipcatcher-ops7"
    names: list[str] = []
    for env_name, suffix in zip(_NAMES, _DEFAULTS, strict=True):
        names.append(os.environ.get(env_name, "").strip() or f"{project}-{suffix}")
    return names


def _health(container: str) -> str:
    try:
        proc = subprocess.run(
            [
                *docker_argv(),
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{end}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _request(url: str, ca_file: Path) -> tuple[int, dict[str, str]]:
    context = ssl.create_default_context(cafile=str(ca_file))
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, context=context, timeout=10) as response:
            response.read()
            return response.status, {key: value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, {key: value for key, value in exc.headers.items()}
    except (OSError, urllib.error.URLError, TimeoutError):
        return 0, {}


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        _emit({"error": "prod-rollback gate takes no arguments", "ok": False})
        return 1
    tls = Path(os.environ.get("DIPCATCHER_TLS_DIR", "/tmp/dipcatcher-ops7-tls"))
    ca_file = tls / "dev.crt"
    if not ca_file.is_file():
        _emit({"error": "dev.crt missing", "ok": False})
        return 1
    lab_port = os.environ.get("LAB_TLS_PORT", "8443").strip() or "8443"
    research_port = os.environ.get("RESEARCH_TLS_PORT", "8444").strip() or "8444"
    containers = _containers()
    deadline = time.monotonic() + _TIMEOUT_S
    states = ["" for _ in containers]
    lab_ok = False
    research_ok = False
    while True:
        states = [_health(name) for name in containers]
        lab_status, lab_headers = _request(f"https://127.0.0.1:{lab_port}/health", ca_file)
        research_status, research_headers = _request(
            f"https://127.0.0.1:{research_port}/health", ca_file
        )
        lab_ok = listener_ok(lab_status, lab_headers)
        research_ok = listener_ok(research_status, research_headers)
        if gate_ready(states, lab_ok, research_ok):
            _emit(
                {
                    "lab": lab_status,
                    "ok": True,
                    "research": research_status,
                    "states": states,
                }
            )
            return 0
        if time.monotonic() >= deadline:
            _emit(
                {
                    "lab": lab_status,
                    "ok": False,
                    "research": research_status,
                    "states": states,
                }
            )
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

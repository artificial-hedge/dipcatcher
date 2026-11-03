#!/usr/bin/env python3
"""Start one lab reader, probe it, then roll the stack or abort.

``lab-canary`` is a compose profile. It is not in the proxy pool and it
does not take the write lock. A passing probe recreates ``lab-a``,
``lab-b``, ``research-a``, ``research-b``, and ``proxy`` from the current
image tags. A failing probe leaves those containers as they are. Either
way the canary container is removed. This does not build images and it
does not run ``prod-smoke``. Stdout is one JSON object. API keys are
required and are not printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

_COMPOSE = Path(__file__).resolve().parents[1] / "deploy" / "compose.prod.yml"
CANARY_SERVICE = "lab-canary"
ROLL_SERVICES = ("lab-a", "lab-b", "research-a", "research-b", "proxy")
_KEY_ERROR = "QUANT_API_KEY and RESEARCH_API_KEY are required"
_SMOKE = (
    "import json, urllib.error, urllib.request\n"
    "def status(method, size):\n"
    "    data = None if size is None else b'x' * size\n"
    "    req = urllib.request.Request('http://127.0.0.1:8000/health', data=data, method=method)\n"
    "    try:\n"
    "        with urllib.request.urlopen(req, timeout=5) as resp:\n"
    "            return resp.status\n"
    "    except urllib.error.HTTPError as exc:\n"
    "        return exc.code\n"
    "print(json.dumps({'health': status('GET', None), "
    "'small': status('POST', 100), 'large': status('POST', 65537)}))\n"
)
Runner = Callable[[Sequence[str]], object]
Sleeper = Callable[[float], None]


def decide(smoke_ok: bool) -> str:
    """``roll`` after a passing probe, otherwise ``abort``."""
    return "roll" if smoke_ok else "abort"


def smoke_passed(checks: Mapping[str, object]) -> bool:
    """Health is 200, a small POST is accepted, and 64 KiB plus one byte is 413."""
    health = checks.get("health")
    small = checks.get("small")
    large = checks.get("large")
    if isinstance(health, bool) or isinstance(small, bool) or isinstance(large, bool):
        return False
    if not isinstance(health, int) or not isinstance(small, int) or not isinstance(large, int):
        return False
    return health == 200 and small != 413 and 0 < small < 500 and large == 413


def project_name() -> str:
    for line in _COMPOSE.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    return "dipcatcher-ops7"


def docker_argv(env: Mapping[str, str]) -> list[str]:
    override = env.get("DIPCATCHER_DOCKER", "").strip()
    if override:
        return shlex.split(override)
    sudo = env.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    preserve = env.get("DIPCATCHER_COMPOSE_PRESERVE", "").strip()
    if preserve:
        return [*shlex.split(sudo), f"--preserve-env={preserve}", "docker"]
    return [*shlex.split(sudo), "docker"]


def compose_base(env: Mapping[str, str]) -> list[str]:
    return [*docker_argv(env), "compose", "-f", str(_COMPOSE)]


def parse_checks(stdout: str) -> dict[str, int] | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    found: dict[str, int] = {}
    for key in ("health", "small", "large"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        found[key] = value
    return found


def _code(result: object) -> int:
    code = getattr(result, "returncode", 1)
    return code if isinstance(code, int) else 1


def _text(result: object) -> str:
    stdout = getattr(result, "stdout", "")
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", "replace")
    return stdout if isinstance(stdout, str) else ""


def _probe(
    env: Mapping[str, str], run: Runner, sleep: Sleeper, attempts: int
) -> dict[str, int] | None:
    container = f"{project_name()}-{CANARY_SERVICE}-1"
    docker = docker_argv(env)
    inspect = [
        *docker,
        "inspect",
        "-f",
        "{{if .State.Health}}{{.State.Health.Status}}{{end}}",
        container,
    ]
    healthy = False
    for _attempt in range(attempts):
        inspected = run(inspect)
        if _code(inspected) == 0 and _text(inspected).strip() == "healthy":
            healthy = True
            break
        sleep(1)
    if not healthy:
        return None
    ran = run([*docker, "exec", "-i", container, "python", "-c", _SMOKE])
    if _code(ran) != 0:
        return None
    return parse_checks(_text(ran))


def _keys_ok(env: Mapping[str, str]) -> bool:
    return bool(env.get("QUANT_API_KEY", "").strip() and env.get("RESEARCH_API_KEY", "").strip())


def execute(
    env: Mapping[str, str],
    run: Runner,
    sleep: Sleeper | None = None,
    attempts: int = 60,
) -> dict[str, object]:
    """Bring the canary up, probe it, roll or abort, then remove it."""
    if not _keys_ok(env):
        return {"error": _KEY_ERROR, "ok": False}
    pause = sleep or (lambda _seconds: None)
    base = compose_base(env)
    report: dict[str, object] = {
        "action": "abort",
        "checks": {},
        "ok": False,
        "removed": False,
    }
    started = run(
        [*base, "--profile", "canary", "up", "-d", "--no-deps", "--no-build", CANARY_SERVICE]
    )
    if _code(started) != 0:
        report["error"] = "canary up"
        return report
    try:
        checks = _probe(env, run, pause, attempts)
        passed = checks is not None and smoke_passed(checks)
        action = decide(passed)
        report["action"] = action
        report["checks"] = checks or {}
        if action == "roll":
            rolled = run([*base, "up", "-d", "--no-build", "--force-recreate", *ROLL_SERVICES])
            report["ok"] = _code(rolled) == 0
        return report
    finally:
        removed = run([*base, "--profile", "canary", "rm", "-sf", CANARY_SERVICE])
        report["removed"] = _code(removed) == 0


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Canary one lab replica, then roll or abort").parse_args(
        argv
    )
    report = execute(os.environ, _run)
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

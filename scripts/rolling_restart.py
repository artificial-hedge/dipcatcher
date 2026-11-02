#!/usr/bin/env python3
"""Drain one API container, then start it again.

Steps are SIGTERM, wait until ``/ready`` is not 200, ``docker stop``,
``docker start``, then wait until ``/ready`` is 200. ``--load-url`` runs
``scripts/http_load.py`` across that window and exits 1 when the report
contains an HTTP 5xx.

``/health`` is the load target. ``/ready`` returns 503 while the drain
latch is set, and that 503 is the not-ready signal rather than a crash.
A single replica drops connections while it is stopped. Those transport
errors are not HTTP 5xx. The API key is read from an env file and is not
printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

_LOAD = Path(__file__).resolve().parent / "http_load.py"
_DRAIN_OK = frozenset({"503", "transport"})


def drain_observed(status: str) -> bool:
    """True when readiness has left 200 for a drain or a closed socket."""
    return status in _DRAIN_OK


def load_has_5xx(report: dict[str, object]) -> bool:
    count = report.get("http_5xx", 1)
    return not isinstance(count, int) or count != 0


def _docker(prefix: list[str], args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*prefix, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _ready_status(url: str, key: str, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    if key:
        request.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return str(response.status)
    except urllib.error.HTTPError as exc:
        return str(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return "transport"


def _log(step: str, **fields: object) -> None:
    sys.stderr.write(json.dumps({"step": step, **fields}, sort_keys=True) + "\n")


def _poll_status(
    url: str,
    key: str,
    timeout_s: float,
    *,
    accept: Callable[[str], bool],
) -> str:
    deadline = time.monotonic() + timeout_s
    last = "timeout"
    while True:
        last = _ready_status(url, key, timeout=2.0)
        if accept(last):
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(0.25)


def _wait_stopped(prefix: list[str], container: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        result = _docker(prefix, ["inspect", "-f", "{{.State.Running}}", container])
        if result.returncode == 0 and result.stdout.strip() == "false":
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _load_key(env_file: str | None, key_env: str | None) -> str:
    if not key_env:
        return ""
    found = os.environ.get(key_env, "")
    if found:
        return found
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, value = text.split("=", 1)
            if name == key_env:
                return value
    return ""


def _start_load(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if not args.load_url:
        return None
    command = [
        sys.executable,
        str(_LOAD),
        "--url",
        args.load_url,
        "--concurrency",
        str(args.load_concurrency),
        "--seconds",
        str(args.load_seconds),
    ]
    if args.env_file:
        command.extend(["--env-file", args.env_file])
    if args.key_env:
        command.extend(["--key-env", args.key_env])
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish_load(process: subprocess.Popen[str] | None) -> dict[str, object] | None:
    if process is None:
        return None
    stdout, stderr = process.communicate()
    line = stdout.strip().splitlines()
    if process.returncode != 0 or not line:
        return {"error": "http_load failed", "http_5xx": 1, "detail": stderr[-400:]}
    try:
        report = json.loads(line[-1])
    except json.JSONDecodeError:
        return {"error": "http_load output was not JSON", "http_5xx": 1}
    if not isinstance(report, dict):
        return {"error": "http_load output was not an object", "http_5xx": 1}
    return report


def restart(args: argparse.Namespace, prefix: list[str], key: str) -> dict[str, object]:
    """Run drain, not-ready, stop, start, and ready. Return the step record."""
    steps: list[dict[str, object]] = []
    killed = _docker(prefix, ["kill", "--signal=TERM", args.container])
    if killed.returncode != 0:
        steps.append({"step": "drain", "ok": False, "detail": killed.stderr.strip()[-300:]})
        return {"ok": False, "steps": steps}
    steps.append({"step": "drain", "ok": True, "signal": "SIGTERM"})
    _log("drain", signal="SIGTERM")

    status = _poll_status(args.ready_url, key, args.not_ready_timeout, accept=drain_observed)
    ready_ok = drain_observed(status)
    steps.append({"step": "not_ready", "ok": ready_ok, "status": status})
    _log("not_ready", status=status, ok=ready_ok)
    if not ready_ok:
        return {"ok": False, "steps": steps}

    stopped = _docker(prefix, ["stop", "--time", str(args.stop_timeout), args.container])
    if stopped.returncode != 0:
        steps.append({"step": "stop", "ok": False, "detail": stopped.stderr.strip()[-300:]})
        return {"ok": False, "steps": steps}
    quiet = _wait_stopped(prefix, args.container, args.stop_timeout + 5)
    steps.append({"step": "stop", "ok": quiet})
    _log("stop", ok=quiet)
    if not quiet:
        return {"ok": False, "steps": steps}

    started = _docker(prefix, ["start", args.container])
    if started.returncode != 0:
        steps.append({"step": "start", "ok": False, "detail": started.stderr.strip()[-300:]})
        return {"ok": False, "steps": steps}
    steps.append({"step": "start", "ok": True})
    _log("start")

    def _is_ready(item: str) -> bool:
        return item == "200"

    final = _poll_status(
        args.ready_url,
        key,
        args.ready_timeout,
        accept=_is_ready,
    )
    final_ok = final == "200"
    steps.append({"step": "ready", "ok": final_ok, "status": final})
    _log("ready", status=final, ok=final_ok)
    return {"ok": final_ok, "steps": steps}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drain and restart one API container")
    parser.add_argument("--container", required=True)
    parser.add_argument("--ready-url", required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--key-env")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--load-url")
    parser.add_argument("--load-seconds", type=float, default=45)
    parser.add_argument("--load-concurrency", type=int, default=10)
    parser.add_argument("--not-ready-timeout", type=float, default=20)
    parser.add_argument("--stop-timeout", type=int, default=40)
    parser.add_argument("--ready-timeout", type=float, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    prefix = shlex.split(args.docker)
    key = _load_key(args.env_file, args.key_env)
    load = _start_load(args)
    if load is not None:
        time.sleep(1.0)
    try:
        record = restart(args, prefix, key)
    finally:
        report = _finish_load(load)
    record["load"] = report
    failed = not bool(record["ok"])
    if report is not None and load_has_5xx(report):
        failed = True
        record["http_5xx"] = report.get("http_5xx")
    elif report is not None:
        record["http_5xx"] = report.get("http_5xx")
    sys.stdout.write(json.dumps(record, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

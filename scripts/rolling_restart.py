#!/usr/bin/env python3
"""Restart API replicas without dropping client connections.

``--container`` is the one-replica path. It sends SIGTERM, waits until
``/ready`` is not 200, stops, starts, and waits until ``/ready`` is 200.
A single replica drops connections while it is stopped. Those transport
errors are not HTTP 5xx. The script exits 1 when the load report contains
an HTTP 5xx. ``--zero-drops`` also fails the run when the load counted a
transport error.

``--containers`` cycles replicas one at a time. The default signal is
SIGUSR1, which marks ``/ready`` 503 and leaves the socket open. The script
waits for that 503, waits ``--settle-seconds`` so the proxy health check
can remove the replica, then stops and starts it. It waits until ``/ready``
is 200 and settles again before touching the next replica. The ready probe
runs inside the container and reads the API key from the container
environment. The key is not passed on the command line and is not printed.
This path fails when the load report has an HTTP 5xx or a transport error.

``/health`` is the load target. ``/ready`` returns 503 while the drain
latch is set, and that 503 is the not-ready signal rather than a crash.
"""

from __future__ import annotations

import argparse
import json
import math
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
_PROBE = (
    "import os,urllib.error,urllib.request\n"
    "url=os.environ['DIPCATCHER_PROBE_URL']\n"
    "name=os.environ.get('DIPCATCHER_PROBE_KEY_ENV','')\n"
    "key=os.environ.get(name,'') if name else ''\n"
    "req=urllib.request.Request(url)\n"
    "if key:\n"
    "    req.add_header('X-API-Key', key)\n"
    "try:\n"
    "    resp=urllib.request.urlopen(req, timeout=2)\n"
    "    resp.read()\n"
    "    print(resp.status)\n"
    "except urllib.error.HTTPError as exc:\n"
    "    print(exc.code)\n"
    "except Exception:\n"
    "    print('transport')\n"
)


def drain_observed(status: str) -> bool:
    """True when readiness has left 200 for a drain or a closed socket."""
    return status in _DRAIN_OK


def load_has_5xx(report: dict[str, object]) -> bool:
    count = report.get("http_5xx", 1)
    return not isinstance(count, int) or count != 0


def transport_count(report: dict[str, object]) -> int:
    """How many load attempts never got an HTTP status. ``-1`` is unreadable."""
    status = report.get("status")
    if not isinstance(status, dict):
        return -1
    count = status.get("transport", 0)
    if not isinstance(count, int):
        return -1
    return count


def load_has_transport(report: dict[str, object]) -> bool:
    return transport_count(report) != 0


def drops_fail_the_run(containers: list[str] | None, zero_drops: bool) -> bool:
    """Two-replica restarts, and an explicit flag, require zero transport errors."""
    return bool(containers) or zero_drops


def _docker(
    prefix: list[str],
    args: list[str],
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [*prefix, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        detail = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess([*prefix, *args], 124, "", detail)


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


def _poll(status_of: Callable[[], str], timeout_s: float, accept: Callable[[str], bool]) -> str:
    deadline = time.monotonic() + timeout_s
    last = "timeout"
    while True:
        last = status_of()
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
    if args.ca_file:
        command.extend(["--ca-file", args.ca_file])
    if args.insecure:
        command.append("--insecure")
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


def _probe_status(prefix: list[str], container: str, url: str, key_env: str) -> str:
    result = _docker(
        prefix,
        [
            "exec",
            "-e",
            f"DIPCATCHER_PROBE_URL={url}",
            "-e",
            f"DIPCATCHER_PROBE_KEY_ENV={key_env}",
            container,
            "/app/.venv/bin/python",
            "-c",
            _PROBE,
        ],
        timeout=10,
    )
    if result.returncode != 0:
        return "transport"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else "transport"


def _signal_name(raw: str) -> str:
    token = raw.upper().removeprefix("SIG")
    if token not in {"USR1", "TERM"}:
        raise SystemExit("signal must be USR1 or TERM")
    return token


def _apply_load(
    record: dict[str, object],
    report: dict[str, object] | None,
    *,
    zero_drops: bool,
) -> bool:
    failed = not bool(record["ok"])
    record["load"] = report
    if report is None:
        return failed
    record["http_5xx"] = report.get("http_5xx")
    record["transport"] = transport_count(report)
    if load_has_5xx(report):
        failed = True
    if zero_drops and load_has_transport(report):
        failed = True
    return failed


def restart(args: argparse.Namespace, prefix: list[str], key: str) -> dict[str, object]:
    """Run drain, not-ready, stop, start, and ready for one container."""
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


def _drained(signal_name: str, status: str) -> bool:
    if signal_name == "USR1":
        return status == "503"
    return drain_observed(status)


def _restart_one(
    args: argparse.Namespace,
    prefix: list[str],
    container: str,
    signal_name: str,
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    killed = _docker(prefix, ["kill", "--signal", signal_name, container])
    if killed.returncode != 0:
        detail = killed.stderr.strip()[-300:]
        steps.append({"step": "drain", "container": container, "ok": False, "detail": detail})
        return {"ok": False, "steps": steps}
    steps.append({"step": "drain", "container": container, "ok": True, "signal": signal_name})
    _log("drain", container=container, signal=signal_name)

    def _current() -> str:
        return _probe_status(prefix, container, args.probe_url, args.key_env or "")

    status = _poll(_current, args.not_ready_timeout, lambda item: _drained(signal_name, item))
    drained = _drained(signal_name, status)
    steps.append({"step": "not_ready", "container": container, "ok": drained, "status": status})
    _log("not_ready", container=container, status=status, ok=drained)
    if not drained:
        return {"ok": False, "steps": steps}

    time.sleep(args.settle_seconds)
    steps.append({"step": "settle_down", "container": container, "ok": True})
    stopped = _docker(prefix, ["stop", "--time", str(args.stop_timeout), container])
    quiet = stopped.returncode == 0 and _wait_stopped(prefix, container, args.stop_timeout + 5)
    steps.append({"step": "stop", "container": container, "ok": quiet})
    _log("stop", container=container, ok=quiet)
    if not quiet:
        return {"ok": False, "steps": steps}

    started = _docker(prefix, ["start", container])
    if started.returncode != 0:
        detail = started.stderr.strip()[-300:]
        steps.append({"step": "start", "container": container, "ok": False, "detail": detail})
        return {"ok": False, "steps": steps}
    steps.append({"step": "start", "container": container, "ok": True})
    _log("start", container=container)
    final = _poll(_current, args.ready_timeout, lambda item: item == "200")
    ready = final == "200"
    steps.append({"step": "ready", "container": container, "ok": ready, "status": final})
    _log("ready", container=container, status=final, ok=ready)
    if not ready:
        return {"ok": False, "steps": steps}
    time.sleep(args.settle_seconds)
    steps.append({"step": "settle_up", "container": container, "ok": True})
    return {"ok": True, "steps": steps}


def cycle_replicas(args: argparse.Namespace, prefix: list[str]) -> dict[str, object]:
    """Drain and replace each named replica, and never overlap those windows."""
    signal_name = _signal_name(args.signal)
    steps: list[dict[str, object]] = []
    for container in args.containers:
        one = _restart_one(args, prefix, container, signal_name)
        nested = one["steps"]
        if isinstance(nested, list):
            steps.extend(nested)
        if not one["ok"]:
            return {"ok": False, "steps": steps}
    return {"ok": True, "steps": steps}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drain and restart API containers")
    parser.add_argument("--container")
    parser.add_argument("--containers", nargs="+")
    parser.add_argument("--ready-url")
    parser.add_argument("--probe-url")
    parser.add_argument("--signal", default="")
    parser.add_argument("--settle-seconds", type=float, default=5)
    parser.add_argument("--env-file")
    parser.add_argument("--key-env")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--load-url")
    parser.add_argument("--load-seconds", type=float, default=45)
    parser.add_argument("--load-concurrency", type=int, default=10)
    parser.add_argument("--not-ready-timeout", type=float, default=20)
    parser.add_argument("--stop-timeout", type=int, default=40)
    parser.add_argument("--ready-timeout", type=float, default=90)
    parser.add_argument("--ca-file")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--zero-drops", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.container) == bool(args.containers):
        parser.error("pass exactly one of --container or --containers")
    if args.container and not args.ready_url:
        parser.error("--container requires --ready-url")
    if args.containers and not args.probe_url:
        parser.error("--containers requires --probe-url")
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 30:
        parser.error("settle-seconds must be in [0, 30]")
    if not args.signal:
        args.signal = "USR1" if args.containers else "TERM"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    prefix = shlex.split(args.docker)
    key = _load_key(args.env_file, args.key_env)
    load = _start_load(args)
    if load is not None:
        time.sleep(1.0)
    try:
        if args.containers:
            record = cycle_replicas(args, prefix)
        else:
            record = restart(args, prefix, key)
    finally:
        report = _finish_load(load)
    failed = _apply_load(
        record,
        report,
        zero_drops=drops_fail_the_run(args.containers, args.zero_drops),
    )
    sys.stdout.write(json.dumps(record, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fire every Prometheus alert in deploy/ through a local Alertmanager.

A webhook records each notification. The drill waits until every alert
name has fired, checks that a quiet interval adds no notification, then
clears the series and waits until each fingerprint resolves. Stdout is
one JSON object. This does not start the lab stack and it does not page
an operator.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ALERTS = _ROOT / "deploy" / "observability" / "alerts.yml"
_SLOS = _ROOT / "deploy" / "observability" / "slo_rules.yml"
_PROM_IMAGE = "prom/prometheus:v2.54.1"
_AM_IMAGE = "prom/alertmanager:v0.27.0"
_PROM_NAME = "dipcatcher-ops13-prometheus"
_AM_NAME = "dipcatcher-ops13-alertmanager"
_PROM_PORT = 19090
_AM_PORT = 19093
_FIRE_S = 480.0
_RESOLVE_S = 450.0
_SILENCE_S = 20.0
_ALERT = re.compile(r"(?m)^[ ]*- alert: ([A-Za-z0-9]+)\s*$")
_FS = '{device="/dev/drill",fstype="ext4",mountpoint="/drill"}'
_RO = '{device="/dev/ro",fstype="ext4",mountpoint="/ro"}'
_BUCKETS = ("0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "+Inf")
_REQUIRED = (
    "DipcatcherDiskPressure",
    "DipcatcherKillSwitchBlocking",
    "DipcatcherReconcileDrift",
    "DipcatcherDataStale",
    "DipcatcherBackupAge",
    "DipcatcherCanaryAbort",
)


def alert_names(*texts: str) -> tuple[str, ...]:
    """Alert names in file order, including the SLO file."""
    found: list[str] = []
    for text in texts:
        found.extend(_ALERT.findall(text))
    return tuple(found)


def fault_metrics(n: int) -> str:
    """Series that make every deploy alert true. ``n`` increases per scrape."""
    lines = [
        "dipcatcher_ready 0",
        "dipcatcher_draining 1",
        "dipcatcher_kill_switch_blocking 1",
        "dipcatcher_data_freshness_seconds 90000",
        "dipcatcher_reconcile_drift 1",
        "dipcatcher_backup_last_success_timestamp 0",
        "dipcatcher_canary_abort 1",
        "dipcatcher_split_brain 1",
        "dipcatcher_dependency_down 1",
        "dipcatcher_audit_tamper 1",
        f"node_filesystem_avail_bytes{_FS} 10",
        f"node_filesystem_size_bytes{_FS} 100",
        f"node_filesystem_readonly{_FS} 0",
        f"node_filesystem_readonly{_RO} 1",
        f'dipcatcher_errors_total{{stage="api"}} {n}',
        *_histogram("dipcatcher_stage_latency_seconds", '{stage="api"}', n),
        f'dipcatcher_http_requests_total{{code="500",service="lab"}} {n}',
        *_histogram("dipcatcher_http_request_duration_seconds", '{service="lab"}', n),
    ]
    return "\n".join(lines) + "\n"


def healthy_metrics(now: int, successes: int) -> str:
    """Gauges inside the SLOs, plus a climbing count of successful requests.

    ``successes`` increases on each scrape. ``rate()`` ignores a counter that
    appears once at a constant value, so the success series has to move.
    That keeps the long error-ratio windows fresh and under the burn line.
    Stage counters are omitted so those rates go stale.
    """
    lines = [
        "dipcatcher_ready 1",
        "dipcatcher_draining 0",
        "dipcatcher_kill_switch_blocking 0",
        "dipcatcher_data_freshness_seconds 1",
        "dipcatcher_reconcile_drift 0",
        f"dipcatcher_backup_last_success_timestamp {now}",
        "dipcatcher_canary_abort 0",
        "dipcatcher_split_brain 0",
        "dipcatcher_dependency_down 0",
        "dipcatcher_audit_tamper 0",
        f"node_filesystem_avail_bytes{_FS} 80",
        f"node_filesystem_size_bytes{_FS} 100",
        f"node_filesystem_readonly{_FS} 0",
        f"node_filesystem_readonly{_RO} 0",
        'dipcatcher_http_requests_total{code="500",service="lab"} 0',
        f'dipcatcher_http_requests_total{{code="200",service="lab"}} {successes}',
    ]
    return "\n".join(lines) + "\n"


def notification_verdict(
    names: tuple[str, ...],
    events: list[tuple[str, str, str]],
) -> dict[str, object]:
    """Route, dedupe, and resolve by fingerprint.

    ``events`` are ``(alertname, fingerprint, status)`` in arrival order.
    Dedupe holds when each fingerprint is firing once. Resolve holds when
    each of those fingerprints is later resolved.
    """
    firing: dict[str, int] = {}
    resolved: set[str] = set()
    by_name: dict[str, set[str]] = {name: set() for name in names}
    extras: list[str] = []
    for name, fingerprint, status in events:
        if name not in by_name:
            extras.append(name)
            continue
        if status == "firing":
            firing[fingerprint] = firing.get(fingerprint, 0) + 1
            by_name[name].add(fingerprint)
        elif status == "resolved":
            resolved.add(fingerprint)
    fired_ids = [item for item in firing if firing[item] >= 1]
    deduped = all(count == 1 for count in firing.values()) and not extras
    resolved_ok = all(item in resolved for item in fired_ids) and bool(fired_ids)
    routed = [name for name in names if by_name[name]]
    return {
        "deduped": deduped and len(routed) == len(names),
        "extras": extras,
        "fingerprints": len(fired_ids),
        "missing": [name for name in names if not by_name[name]],
        "resolved": resolved_ok and len(routed) == len(names),
        "routed": routed,
    }


def prometheus_config(fixture: int, empty: int, down: int, am_port: int) -> str:
    """Scrape config for the drill. Rules are the files mounted beside it."""
    return (
        "global:\n"
        "  scrape_interval: 5s\n"
        "  evaluation_interval: 15s\n"
        "rule_files:\n"
        "  - /etc/prometheus/alerts.yml\n"
        "  - /etc/prometheus/slo_rules.yml\n"
        "alerting:\n"
        "  alertmanagers:\n"
        "    - static_configs:\n"
        f"        - targets: ['127.0.0.1:{am_port}']\n"
        "scrape_configs:\n"
        "  - job_name: lab-api\n"
        "    static_configs:\n"
        f"      - targets: ['127.0.0.1:{fixture}', '127.0.0.1:{down}']\n"
        "  - job_name: research-api\n"
        "    static_configs:\n"
        f"      - targets: ['127.0.0.1:{empty}']\n"
    )


def alertmanager_config(webhook: int) -> str:
    """One receiver. Repeats wait four hours so a stable firing stays quiet."""
    return (
        "route:\n"
        "  receiver: drill\n"
        "  group_by: [alertname]\n"
        "  group_wait: 1s\n"
        "  group_interval: 1s\n"
        "  repeat_interval: 4h\n"
        "receivers:\n"
        "  - name: drill\n"
        "    webhook_configs:\n"
        f"      - url: http://127.0.0.1:{webhook}/hook\n"
        "        send_resolved: true\n"
    )


def _histogram(name: str, labels: str, n: int) -> list[str]:
    """``labels`` is a brace-wrapped selector such as ``{stage="api"}``."""
    inner = labels[1:-1]
    lines = [
        f'{name}_bucket{{{inner},le="{bound}"}} {n if bound == "+Inf" else 0}' for bound in _BUCKETS
    ]
    lines.append(f"{name}_sum{{{inner}}} {n * 3}")
    lines.append(f"{name}_count{{{inner}}} {n}")
    return lines


class _Metrics:
    def __init__(self) -> None:
        self.mode = "fault"
        self.n = 0
        self.successes = 0
        self.lock = threading.Lock()

    def body(self) -> bytes:
        with self.lock:
            self.n += 1
            n = self.n
            mode = self.mode
            if mode != "fault":
                self.successes += 100_000
            successes = self.successes
        if mode == "fault":
            return fault_metrics(n).encode()
        return healthy_metrics(int(time.time()), successes).encode()


class _Events:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.lock = threading.Lock()

    def add(self, row: tuple[str, str, str]) -> None:
        with self.lock:
            self.rows.append(row)

    def snapshot(self) -> list[tuple[str, str, str]]:
        with self.lock:
            return list(self.rows)


def _handler(metrics: _Metrics, events: _Events, kind: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            if kind == "empty":
                payload = b""
            else:
                payload = metrics.body()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""
            self._record(raw)
            self.send_response(204)
            self.end_headers()

        def _record(self, raw: bytes) -> None:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                return
            alerts = payload.get("alerts") if isinstance(payload, dict) else None
            if not isinstance(alerts, list):
                return
            for item in alerts:
                if not isinstance(item, dict):
                    continue
                labels = item.get("labels")
                name = labels.get("alertname") if isinstance(labels, dict) else None
                fingerprint = str(item.get("fingerprint") or "")
                status = str(item.get("status") or "")
                if isinstance(name, str) and fingerprint and status:
                    events.add((name, fingerprint, status))

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, name="alert-drill", daemon=True).start()
    return server


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _run(argv: list[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _rm(name: str) -> None:
    _run([*_docker(), "rm", "-f", name], timeout=30)


def _ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/-/ready", timeout=2) as response:
            return int(response.status) == 200
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _wait_ready(port: int) -> bool:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _ready(port):
            return True
        time.sleep(0.5)
    return False


def _states() -> list[str]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{_PROM_PORT}/api/v1/alerts", timeout=5
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, UnicodeError, json.JSONDecodeError):
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    alerts = data.get("alerts") if isinstance(data, dict) else None
    if not isinstance(alerts, list):
        return []
    found: list[str] = []
    for item in alerts:
        if not isinstance(item, dict):
            continue
        labels = item.get("labels")
        name = labels.get("alertname") if isinstance(labels, dict) else ""
        found.append(f"{name}:{item.get('state')}")
    return sorted(found)


def _covered(names: tuple[str, ...], events: list[tuple[str, str, str]], status: str) -> bool:
    seen = {name for name, _fingerprint, kind in events if kind == status}
    return all(name in seen for name in names)


def _quiet(before: int, events: _Events) -> bool:
    time.sleep(_SILENCE_S)
    return len(events.snapshot()) == before


def _start_containers(work: Path) -> str | None:
    docker = _docker()
    _rm(_PROM_NAME)
    _rm(_AM_NAME)
    prom = _run(
        [
            *docker,
            "run",
            "-d",
            "--name",
            _PROM_NAME,
            "--network",
            "host",
            "-v",
            f"{work / 'prometheus.yml'}:/etc/prometheus/prometheus.yml:ro",
            "-v",
            f"{_ALERTS}:/etc/prometheus/alerts.yml:ro",
            "-v",
            f"{_SLOS}:/etc/prometheus/slo_rules.yml:ro",
            _PROM_IMAGE,
            f"--web.listen-address=127.0.0.1:{_PROM_PORT}",
            "--config.file=/etc/prometheus/prometheus.yml",
            "--storage.tsdb.path=/prometheus",
            "--storage.tsdb.retention.time=2h",
        ]
    )
    if prom.returncode != 0:
        return "prometheus"
    am = _run(
        [
            *docker,
            "run",
            "-d",
            "--name",
            _AM_NAME,
            "--network",
            "host",
            "-v",
            f"{work / 'alertmanager.yml'}:/etc/alertmanager/alertmanager.yml:ro",
            _AM_IMAGE,
            "--config.file=/etc/alertmanager/alertmanager.yml",
            f"--web.listen-address=127.0.0.1:{_AM_PORT}",
            "--cluster.listen-address=",
        ]
    )
    if am.returncode != 0:
        return "alertmanager"
    if not _wait_ready(_PROM_PORT) or not _wait_ready(_AM_PORT):
        return "ready"
    return None


def _check_rules() -> bool:
    proc = _run(
        [
            *_docker(),
            "run",
            "--rm",
            "--entrypoint",
            "promtool",
            "-v",
            f"{_ALERTS}:/alerts.yml:ro",
            "-v",
            f"{_SLOS}:/slo.yml:ro",
            _PROM_IMAGE,
            "check",
            "rules",
            "/alerts.yml",
            "/slo.yml",
        ],
        timeout=120,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode == 0


def execute() -> dict[str, object]:
    """Start Prometheus and Alertmanager, fire the rules, then resolve them."""
    names = alert_names(_ALERTS.read_text(encoding="utf-8"), _SLOS.read_text(encoding="utf-8"))
    missing = [name for name in _REQUIRED if name not in names]
    if missing or not _check_rules():
        return {"error": "rules", "missing_required": missing, "ok": False}
    metrics = _Metrics()
    events = _Events()
    fixture = _serve(_handler(metrics, events, "metrics"))
    empty = _serve(_handler(metrics, events, "empty"))
    work = Path("/tmp/dipcatcher-ops13-alert")
    work.mkdir(parents=True, exist_ok=True)
    down_port = _free_port()
    (work / "prometheus.yml").write_text(
        prometheus_config(
            int(fixture.server_address[1]),
            int(empty.server_address[1]),
            down_port,
            _AM_PORT,
        ),
        encoding="utf-8",
    )
    hook = _serve(_handler(metrics, events, "hook"))
    (work / "alertmanager.yml").write_text(
        alertmanager_config(int(hook.server_address[1])),
        encoding="utf-8",
    )
    started = time.monotonic()
    try:
        failure = _start_containers(work)
        if failure:
            _dump_logs()
            return {"error": failure, "ok": False, "states": _states()}
        if not _poll_names(names, events, _FIRE_S):
            _dump_logs()
            return _report(names, events, started, False, "firing")
        silence_ok = _quiet(len(events.snapshot()), events)
        metrics.mode = "healthy"
        down = _serve_port(_handler(metrics, events, "empty"), down_port)
        resolved = _poll_resolved(names, events, _RESOLVE_S)
        verdict = notification_verdict(names, events.snapshot())
        ok = bool(verdict["deduped"] and verdict["resolved"] and silence_ok and resolved)
        report = _report(names, events, started, ok, "")
        report["silence_ok"] = silence_ok
        report["verdict"] = verdict
        down.shutdown()
        return report
    finally:
        _rm(_PROM_NAME)
        _rm(_AM_NAME)
        fixture.shutdown()
        empty.shutdown()
        hook.shutdown()


def _poll_names(names: tuple[str, ...], events: _Events, limit: float) -> bool:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if _covered(names, events.snapshot(), "firing"):
            return True
        seen = {row[0] for row in events.snapshot()}
        sys.stderr.write(f"firing {len(seen)} / {len(names)} states={','.join(_states())}\n")
        sys.stderr.flush()
        time.sleep(5)
    return False


def _poll_resolved(names: tuple[str, ...], events: _Events, limit: float) -> bool:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if _covered(names, events.snapshot(), "resolved"):
            return True
        seen = {row[0] for row in events.snapshot() if row[2] == "resolved"}
        sys.stderr.write(f"resolved {len(seen)} / {len(names)} states={','.join(_states())}\n")
        sys.stderr.flush()
        time.sleep(5)
    return False


def _report(
    names: tuple[str, ...],
    events: _Events,
    started: float,
    ok: bool,
    error: str,
) -> dict[str, object]:
    rows = events.snapshot()
    verdict = notification_verdict(names, rows)
    return {
        "alerts": list(names),
        "error": error,
        "fingerprints": verdict["fingerprints"],
        "missing": verdict["missing"],
        "ok": ok,
        "seconds": round(time.monotonic() - started, 3),
        "states": _states(),
        "webhook_posts": len(rows),
    }


def _dump_logs() -> None:
    for name in (_PROM_NAME, _AM_NAME):
        proc = _run([*_docker(), "logs", "--tail", "60", name], timeout=20)
        sys.stderr.write(f"--- {name}\n")
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])


def _free_port() -> int:
    import socket  # noqa: PLC0415

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _serve_port(handler: type[BaseHTTPRequestHandler], port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, name="alert-down", daemon=True).start()
    return server


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-alert-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/usr/bin/env python3
"""Hold the lab proxy under synthetic load and sample RSS and file descriptors.

Default duration is 1200 seconds. ``/health`` is outside the rate limit.
Stdout is one JSON object with p50, p99, the error rate, and the slopes.
A positive file-descriptor slope or a large RSS slope fails the process.
The API key is not printed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SECONDS = 1200.0
_WARMUP_S = 120.0
_FD_PER_S = 0.05
_RSS_KB_PER_S = 20.0
_CONTAINERS = ("lab-a", "lab-b", "proxy")


def slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of ``y`` against ``x``. Zero when the fit is undefined."""
    count = len(points)
    if count < 2:
        return 0.0
    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    numerator = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def leak_decision(
    samples: list[dict[str, float]], *, warmup_s: float = _WARMUP_S
) -> dict[str, object]:
    """Flag a file-descriptor climb or an RSS climb after ``warmup_s``."""
    if not samples:
        return {"fd_per_s": 0.0, "leak": True, "rss_kb_per_s": 0.0}
    origin = samples[0]["t"]
    kept = [item for item in samples if item["t"] - origin >= warmup_s]
    if len(kept) < 3:
        kept = samples
    fd_slope = slope([(item["t"], item["fds"]) for item in kept])
    rss_slope = slope([(item["t"], item["rss_kb"]) for item in kept])
    return {
        "fd_per_s": fd_slope,
        "leak": fd_slope > _FD_PER_S or rss_slope > _RSS_KB_PER_S,
        "rss_kb_per_s": rss_slope,
        "samples": len(kept),
    }


def _http_load() -> object:
    path = Path(__file__).resolve().parent / "http_load.py"
    name = "dipcatcher_http_load_soak"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("http_load")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _container(leaf: str) -> str:
    project = os.environ.get("PROJECT", "dipcatcher-ops7")
    return f"{project}-{leaf}-1"


def _sample(container: str) -> tuple[int, int] | None:
    script = "awk '/VmRSS/ {print $2}' /proc/1/status; ls -1 /proc/1/fd | wc -l"
    proc = subprocess.run(
        [*_docker(), "exec", container, "sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 2 or not lines[0].isdigit() or not lines[1].isdigit():
        return None
    return int(lines[0]), int(lines[1])


def _collect(container: str, samples: list[dict[str, float]], stop: threading.Event) -> None:
    started = time.monotonic()
    while not stop.is_set():
        reading = _sample(container)
        if reading is not None:
            rss_kb, fds = reading
            samples.append(
                {"fds": float(fds), "rss_kb": float(rss_kb), "t": time.monotonic() - started}
            )
        stop.wait(30)


def _health(url: str, ca: str) -> bool:
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=5, cafile=ca) as response:
            return int(response.status) == 200
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _up() -> int:
    env = os.environ.copy()
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(name, None)
    proc = subprocess.run(
        ["make", "-f", "deploy/prod.mk", "prod-up"],
        cwd=_ROOT,
        env=env,
        timeout=900,
        check=False,
    )
    return proc.returncode


def _capture_evidence() -> None:
    """Copy logs, metrics, and Prometheus alerts before the stack stops.

    The metrics request reads the key inside the container. The key is not
    an argument and is not written to the capture files on purpose.
    """
    log_dir = Path(os.environ.get("DIPCATCHER_LOG_DIR", "/tmp/dipcatcher-ops7-logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    prom_port = os.environ.get("PROMETHEUS_PUBLISH_PORT", "9090").strip() or "9090"
    _save_http(f"http://127.0.0.1:{prom_port}/api/v1/alerts", log_dir / "alerts.json")
    _save_http(f"http://127.0.0.1:{prom_port}/api/v1/rules", log_dir / "alert-rules.json")
    leaves = ("lab-a", "lab-b", "research-a", "research-b", "proxy", "prometheus")
    for leaf in leaves:
        _save_logs(_container(leaf), log_dir / f"docker-{leaf}.log")
    _save_metrics("lab-a", "/run/secrets/quant_api_key", "/logs/metrics-lab-a.prom")
    _save_metrics("research-a", "/run/secrets/research_api_key", "/logs/metrics-research-a.prom")


def _write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload``. The log directory is owned by uid 10001."""
    try:
        path.write_bytes(payload)
        return
    except OSError:
        pass
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    prefix = shlex.split(sudo.strip()) if sudo.strip() else []
    if not prefix:
        return
    tmp = Path("/tmp") / f"dipcatcher-capture-{path.name}"
    try:
        tmp.write_bytes(payload)
        subprocess.run([*prefix, "cp", str(tmp), str(path)], check=False, timeout=30)
        subprocess.run([*prefix, "chmod", "644", str(path)], check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return
    finally:
        tmp.unlink(missing_ok=True)


def _save_http(url: str, path: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError, TimeoutError):
        return
    _write_bytes(path, payload)


def _save_logs(container: str, path: Path) -> None:
    try:
        proc = subprocess.run(
            [*_docker(), "logs", container],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    _write_bytes(path, proc.stdout + proc.stderr)


def _save_metrics(leaf: str, secret_path: str, output_path: str) -> None:
    script = (
        "import os, pathlib, urllib.request\n"
        "token = pathlib.Path(os.environ['DIPCATCHER_CAPTURE_SECRET']).read_text().splitlines()[0]\n"
        "req = urllib.request.Request('http://127.0.0.1:9464/metrics',"
        " headers={'Authorization': 'Bearer ' + token})\n"
        "body = urllib.request.urlopen(req, timeout=5).read()\n"
        "pathlib.Path(os.environ['DIPCATCHER_CAPTURE_OUT']).write_bytes(body)\n"
    )
    try:
        subprocess.run(
            [
                *_docker(),
                "exec",
                "-e",
                f"DIPCATCHER_CAPTURE_SECRET={secret_path}",
                "-e",
                f"DIPCATCHER_CAPTURE_OUT={output_path}",
                _container(leaf),
                "python",
                "-c",
                script,
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _down() -> None:
    env = os.environ.copy()
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(name, None)
    subprocess.run(
        ["make", "-f", "deploy/prod.mk", "prod-down"],
        cwd=_ROOT,
        env=env,
        timeout=180,
        check=False,
    )


def execute() -> dict[str, object]:
    """Bring the stack up when health is down, load it, and sample lab-a."""
    seconds = float(os.environ.get("DIPCATCHER_SOAK_SECONDS", str(_DEFAULT_SECONDS)))
    if seconds <= 0 or seconds > _DEFAULT_SECONDS:
        return {"error": "seconds", "ok": False}
    port = os.environ.get("LAB_TLS_PORT", "8443")
    tls = os.environ.get("DIPCATCHER_TLS_DIR", "/tmp/dipcatcher-ops7-tls")
    url = f"https://127.0.0.1:{port}/health"
    ca = str(Path(tls) / "dev.crt")
    started_here = False
    if not _health(url, ca):
        if _up() != 0:
            return {"error": "prod-up", "ok": False}
        started_here = True
    container = _container("lab-a")
    samples: list[dict[str, float]] = []
    stop = threading.Event()
    watcher = threading.Thread(target=_collect, args=(container, samples, stop), daemon=True)
    watcher.start()
    module = _http_load()
    host, number, path, https = module._parse_url(url)
    context = module._client_ssl(https, ca, False)
    payload = module._request_bytes(host, path, "")
    peers: dict[str, bool] = {}
    try:
        report = asyncio.run(module._run(host, number, payload, 20, seconds, context))
        peers = {leaf: _sample(_container(leaf)) is not None for leaf in _CONTAINERS}
    finally:
        stop.set()
        watcher.join(timeout=5)
        if started_here:
            _capture_evidence()
            _down()
    decision = leak_decision(samples)
    ok = not decision["leak"] and report.get("error_rate") == 0 and report.get("requests", 0) > 0
    return {
        "container": container,
        "error_rate": report.get("error_rate"),
        "fd_per_s": decision["fd_per_s"],
        "leak": decision["leak"],
        "ok": ok,
        "p50_ms": report.get("p50_ms"),
        "p99_ms": report.get("p99_ms"),
        "peers_seen": peers,
        "requests": report.get("requests"),
        "rss_kb_per_s": decision["rss_kb_per_s"],
        "samples": len(samples),
        "seconds": seconds,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-soak takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

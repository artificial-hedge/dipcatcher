#!/usr/bin/env python3
"""Check the production host before compose up.

Stdout is one JSON object. The process exits 0 only when every check
passed, and 1 otherwise. API key values are not part of the JSON.
``prod-up`` runs this target after certificates and data directories exist
and before it builds images.
"""

from __future__ import annotations

import calendar
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = _ROOT / "deploy" / "compose.prod.yml"
_PROD_ENV = _ROOT / "deploy" / "prod.env"
_MIN_FREE = 64 * 1024 * 1024
_MIN_REMAINING_S = 86400
_MAX_SKEW_S = 120
_MIN_YEAR = 2024
_MAX_YEAR = 2035
_PUBLISH = (
    ("lab", "LAB_TLS_PORT", 8443),
    ("research", "RESEARCH_TLS_PORT", 8444),
    ("prometheus", "PROMETHEUS_PUBLISH_PORT", 9090),
    ("grafana", "GRAFANA_PUBLISH_PORT", 3000),
)
_FORBIDDEN_PUBLISH = 9464


def clock_ok(now: float, not_before: float, not_after: float) -> bool:
    """True when the clock is in range and the certificate still has a day."""
    year = time.gmtime(now).tm_year
    if year < _MIN_YEAR or year > _MAX_YEAR:
        return False
    if now + _MAX_SKEW_S < not_before:
        return False
    if now > not_after:
        return False
    return not_after - now >= _MIN_REMAINING_S


def port_free(port: int) -> bool:
    """True when nothing is listening on loopback ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _single_token(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False
    return not any(char in value for char in "\"'{}#")


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def _check(name: str, ok: bool, detail: str = "") -> dict[str, object]:
    item: dict[str, object] = {"name": name, "ok": ok}
    if detail:
        item["detail"] = detail
    return item


def check_env() -> dict[str, object]:
    """Require both API keys as single tokens. Do not record the values."""
    missing = [
        name
        for name in ("QUANT_API_KEY", "RESEARCH_API_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        return _check("env", False, "missing " + ",".join(missing))
    bad = [
        name
        for name in ("QUANT_API_KEY", "RESEARCH_API_KEY")
        if not _single_token(os.environ.get(name, "").strip())
    ]
    if bad:
        return _check("env", False, "not a single token " + ",".join(bad))
    return _check("env", True)


def check_certs(tls_dir: Path) -> dict[str, object]:
    cert = tls_dir / "dev.crt"
    key = tls_dir / "dev.key"
    if not cert.is_file() or cert.stat().st_size == 0:
        return _check("certs", False, "dev.crt missing")
    if not key.is_file() or key.stat().st_size == 0:
        return _check("certs", False, "dev.key missing")
    if stat.S_IMODE(key.stat().st_mode) & 0o077:
        return _check("certs", False, "dev.key mode")
    return _check("certs", True)


def _port_value(name: str, default: int) -> int | None:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1 or value > 65535:
        return None
    return value


def check_ports() -> dict[str, object]:
    chosen: dict[str, int] = {}
    for label, name, default in _PUBLISH:
        value = _port_value(name, default)
        if value is None:
            return _check("ports", False, f"{name} invalid")
        chosen[label] = value
    if len(set(chosen.values())) != len(chosen):
        return _check("ports", False, "ports are not distinct")
    if _FORBIDDEN_PUBLISH in chosen.values():
        return _check("ports", False, "9464 is not a publish port")
    busy = [label for label, value in chosen.items() if not port_free(value)]
    if busy:
        return _check("ports", False, "busy " + ",".join(busy))
    return _check("ports", True)


def check_disk(paths: list[Path]) -> dict[str, object]:
    short: list[str] = []
    for path in paths:
        if not path.is_dir():
            short.append(path.name or str(path))
            continue
        if shutil.disk_usage(path).free < _MIN_FREE:
            short.append(path.name or str(path))
    if short:
        return _check("disk", False, "low " + ",".join(short))
    return _check("disk", True)


def check_write_owner() -> dict[str, object]:
    """The lab writer is lab-a. Research is read-only and is not a writer."""
    try:
        loaded = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
        env = _PROD_ENV.read_text(encoding="utf-8")
    except (OSError, yaml.YAMLError):
        return _check("write-owner", False, "unreadable")
    services = loaded.get("services", {}) if isinstance(loaded, dict) else {}
    expected = {"lab-a": "1", "lab-b": "0", "research-a": "0", "research-b": "0"}
    for name, owner in expected.items():
        service = services.get(name, {})
        environment = service.get("environment", {}) if isinstance(service, dict) else {}
        if environment.get("DIPCATCHER_WRITE_OWNER") != owner:
            return _check("write-owner", False, name)
        volumes = service.get("volumes", [])
        if not volumes or not isinstance(volumes[0], str):
            return _check("write-owner", False, name)
        read_only = volumes[0].endswith(":/app/data:ro")
        if name.startswith("research") and not read_only:
            return _check("write-owner", False, "research mount")
        if name.startswith("lab") and read_only:
            return _check("write-owner", False, "lab mount")
    if "DIPCATCHER_PROFILE=production" not in env:
        return _check("write-owner", False, "profile")
    if not _prod_rate_ok(env):
        return _check("write-owner", False, "rate")
    if "DIPCATCHER_REQUIRE_DATA_WRITE=1" not in env:
        return _check("write-owner", False, "data write")
    if "API_KEY=" in env:
        return _check("write-owner", False, "secret in prod.env")
    return _check("write-owner", True)


def _prod_rate_ok(env: str) -> bool:
    for line in env.splitlines():
        if not line.startswith("DIPCATCHER_RATE_LIMIT_RPS="):
            continue
        raw = line.split("=", 1)[1].strip()
        try:
            value = float(raw)
        except ValueError:
            return False
        return value > 0.0 and value <= 1_000_000.0
    return False


def _cert_window(cert: Path) -> tuple[float, float] | None:
    start = _openssl_epoch(cert, "-startdate")
    end = _openssl_epoch(cert, "-enddate")
    if start is None or end is None:
        return None
    return start, end


def _openssl_epoch(cert: Path, flag: str) -> float | None:
    try:
        proc = subprocess.run(
            ["openssl", "x509", "-in", str(cert), "-noout", flag],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    raw = proc.stdout.split("=", 1)[-1].strip().removesuffix(" GMT")
    try:
        parsed = time.strptime(raw, "%b %d %H:%M:%S %Y")
    except ValueError:
        return None
    return float(calendar.timegm(parsed))


def check_clock(cert: Path, now: float | None = None) -> dict[str, object]:
    window = _cert_window(cert)
    if window is None:
        return _check("clock", False, "certificate dates")
    moment = time.time() if now is None else now
    if not clock_ok(moment, window[0], window[1]):
        return _check("clock", False, "skew")
    return _check("clock", True)


def collect_checks() -> list[dict[str, object]]:
    tls = Path(os.environ.get("DIPCATCHER_TLS_DIR", "/tmp/dipcatcher-ops7-tls"))
    lab = Path(os.environ.get("DIPCATCHER_LAB_DATA", "/tmp/dipcatcher-ops7-data/lab"))
    research = Path(
        os.environ.get("DIPCATCHER_RESEARCH_DATA", "/tmp/dipcatcher-ops7-data/research")
    )
    return [
        check_env(),
        check_certs(tls),
        check_ports(),
        check_disk([lab, research, tls]),
        check_write_owner(),
        check_clock(tls / "dev.crt"),
    ]


def summarize(checks: list[dict[str, object]]) -> dict[str, object]:
    ok = bool(checks) and all(bool(item.get("ok")) for item in checks)
    return {"checks": checks, "ok": ok}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        _emit({"error": "prod-preflight takes no arguments", "ok": False})
        return 1
    report = summarize(collect_checks())
    _emit(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

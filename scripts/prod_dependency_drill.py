#!/usr/bin/env python3
"""DNS, TLS, broker-down, and slow-broker probes fail closed in paper mode.

Each case writes no intent, pages the dependency gauge, then a healthy
local broker accepts the same order once. Stdout is one JSON object.
Research and simulated paper only.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORK = Path("/tmp/dipcatcher-ops17-deps")
_RULES = _ROOT / "deploy" / "observability"
_URL = "DIPCATCHER_PAPER_BROKER_URL"
_TIMEOUT = "DIPCATCHER_PAPER_BROKER_TIMEOUT"
_CA = "DIPCATCHER_PAPER_BROKER_CA"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        delay = float(getattr(self.server, "delay", 0.0))
        if delay:
            time.sleep(delay)
        body = b"ok"
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _serve(delay: float) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.delay = delay  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _expired(directory: Path) -> tuple[Path, Path, Path, str]:
    """Write a throwaway CA and a leaf that expired in 2020. Files stay under ``directory``."""
    import ipaddress
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    directory.mkdir(parents=True, exist_ok=True)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "dipcatcher-ops17-ca")])
    now = datetime.now(UTC)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    not_after = datetime(2020, 1, 2, tzinfo=UTC)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime(2020, 1, 1, tzinfo=UTC))
        .not_valid_after(not_after)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    key_path = directory / "srv.key"
    cert_path = directory / "srv.crt"
    ca_path = directory / "ca.crt"
    key_path.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path, ca_path, not_after.isoformat()


def _https(cert: Path, key: Path) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _peek_tls(port: int, ca: Path) -> str:
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=str(ca))
    try:
        with (
            socket.create_connection(("127.0.0.1", port), timeout=2) as raw,
            context.wrap_socket(raw, server_hostname="127.0.0.1"),
        ):
            return ""
    except OSError as exc:
        return str(exc)


def _docker() -> list[str]:
    import shlex

    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _promtool() -> dict[str, object]:
    proc = subprocess.run(
        [
            *_docker(),
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "promtool",
            "-v",
            f"{_RULES}:/rules:ro",
            "-w",
            "/rules",
            "prom/prometheus:v2.54.1",
            "test",
            "rules",
            "ops17_guard_test.yml",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    detail = ((proc.stdout or "") + (proc.stderr or ""))[-800:]
    return {"detail": detail, "exit": proc.returncode, "ok": proc.returncode == 0}


def _intents(state: Path, order_id: str) -> int:
    from quant_fund.paper.durable_book import journal_path, read_json_lines

    path = journal_path(state)
    if not path.is_file():
        return 0
    rows = read_json_lines(path)
    return sum(1 for row in rows if row.get("kind") == "intent" and row.get("order_id") == order_id)


def _case(
    name: str, url: str, timeout: str, healthy: str, extra: dict[str, object]
) -> dict[str, object]:
    from quant_fund.api.lifecycle import acceptance_block, reset_runtime_state
    from quant_fund.observe.metrics import render_prometheus
    from quant_fund.paper.durable_book import (
        ledger_order_count,
        ledger_path,
        open_book,
        run_orders,
        submit_once,
    )
    from quant_fund.paper.order_lease import release_order_lease

    state = _WORK / name
    order_id = f"{name}-0"
    os.environ[_URL] = url
    os.environ[_TIMEOUT] = timeout
    failed = run_orders(state, 1, name)
    journal = _intents(state, order_id)
    opened = (state / "broker_state.json").is_file()
    gauge = "dipcatcher_dependency_down 1" in render_prometheus()
    acceptance = acceptance_block()
    os.environ[_URL] = healthy
    os.environ[_TIMEOUT] = "2"
    os.environ.pop(_CA, None)
    recovered = run_orders(state, 1, name)
    duplicate = submit_once(open_book(state), state, order_id)
    count = ledger_order_count(ledger_path(state), order_id)
    cleared = "dipcatcher_dependency_down" not in render_prometheus()
    ok = (
        failed == 1
        and journal == 0
        and not opened
        and gauge
        and recovered == 0
        and duplicate == "duplicate"
        and count == 1
        and _intents(state, order_id) == 1
        and cleared
    )
    release_order_lease(state)
    reset_runtime_state()
    payload: dict[str, object] = {
        "acceptance": acceptance,
        "duplicate": duplicate,
        "failed": failed,
        "gauge": gauge,
        "intents": _intents(state, order_id),
        "intents_during_outage": journal,
        "opened_during_outage": opened,
        "ledger_count": count,
        "ok": ok,
        "recovered": recovered,
    }
    payload.update(extra)
    return payload


def _lookup() -> str:
    try:
        socket.getaddrinfo("dipcatcher-ops17-nx.invalid", 9, type=socket.SOCK_STREAM)
    except OSError as exc:
        return str(exc)
    return ""


def execute() -> dict[str, object]:
    saved = {name: os.environ.get(name) for name in (_URL, _TIMEOUT, _CA)}
    if _WORK.exists():
        shutil.rmtree(_WORK)
    _WORK.mkdir(parents=True)
    healthy_server = _serve(0.0)
    slow_server = _serve(1.5)
    refused = socket.socket()
    refused.bind(("127.0.0.1", 0))
    tls_server: ThreadingHTTPServer | None = None
    try:
        healthy_port = int(healthy_server.server_address[1])
        healthy = f"http://127.0.0.1:{healthy_port}/"
        slow_port = int(slow_server.server_address[1])
        refused_port = int(refused.getsockname()[1])
        cert, key, ca, enddate = _expired(_WORK / "certs")
        tls_server = _https(cert, key)
        tls_port = int(tls_server.server_address[1])
        os.environ[_CA] = str(ca)
        tls_error = _peek_tls(tls_port, ca)
        dns = _case(
            "dns",
            "http://dipcatcher-ops17-nx.invalid:9/",
            "1",
            healthy,
            {"lookup": _lookup()},
        )
        os.environ[_CA] = str(ca)
        tls = _case(
            "tls",
            f"https://127.0.0.1:{tls_port}/",
            "2",
            healthy,
            {"enddate": enddate, "tls_error": tls_error},
        )
        down = _case(
            "down",
            f"http://127.0.0.1:{refused_port}/",
            "1",
            healthy,
            {},
        )
        slow = _case(
            "slow",
            f"http://127.0.0.1:{slow_port}/",
            "0.2",
            healthy,
            {},
        )
        rules = _promtool()
        dns_ok = dns["ok"] is True and dns["acceptance"] == "dependency_dns" and bool(dns["lookup"])
        tls_ok = (
            tls["ok"] is True
            and tls["acceptance"] == "dependency_tls"
            and "expired" in str(tls["tls_error"]).lower()
            and "2020" in str(tls["enddate"])
        )
        down_ok = down["ok"] is True and down["acceptance"] == "dependency_broker_down"
        slow_ok = slow["ok"] is True and slow["acceptance"] == "dependency_broker_slow"
        ok = bool(dns_ok and tls_ok and down_ok and slow_ok and rules["ok"])
        return {
            "broker_down": down,
            "broker_slow": slow,
            "dns": dns,
            "error": "" if ok else "dependency",
            "ok": ok,
            "promtool": rules,
            "tls": tls,
        }
    finally:
        refused.close()
        healthy_server.shutdown()
        slow_server.shutdown()
        if tls_server is not None:
            tls_server.shutdown()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        from quant_fund.api.lifecycle import reset_runtime_state
        from quant_fund.paper.order_lease import release_order_lease

        release_order_lease()
        reset_runtime_state()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-dependency-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

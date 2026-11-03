"""Two-replica proxy, single writer, and zero-drop restart helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from quant_fund.api.app import _MAX_REQUEST_BYTES
from quant_fund.api.lifecycle import is_draining, reset_runtime_state
from quant_fund.api.service_config import ENV_CONTRACT
from quant_fund.paper.ledger import PaperLedger, _atomic_write_text
from quant_fund.utils.operator_kill import OperatorKillError, write_operator_kill
from quant_fund.utils.write_owner import (
    WriteOwnerError,
    claim_write_lock,
    release_write_lock,
    require_write_owner,
    write_owner_mode,
)

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clear_owner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DIPCATCHER_WRITE_OWNER", raising=False)
    monkeypatch.delenv("DIPCATCHER_DATA_ROOT", raising=False)
    release_write_lock()
    yield
    release_write_lock()
    reset_runtime_state()


def test_write_owner_is_outside_the_env_contract() -> None:
    assert "DIPCATCHER_WRITE_OWNER" not in ENV_CONTRACT
    example = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "DIPCATCHER_WRITE_OWNER" not in example
    assert "DIPCATCHER_WRITE_OWNER" in (_ROOT / "docs" / "PRODUCTION_RUNBOOK.md").read_text(
        encoding="utf-8"
    )


def test_unset_owner_still_writes(tmp_path: Path) -> None:
    assert write_owner_mode() == "unset"
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "ENABLED")
    assert path.read_text(encoding="utf-8").strip() == "ENABLED"
    ledger = tmp_path / "note.txt"
    _atomic_write_text(ledger, "ok\n")
    assert ledger.read_text(encoding="utf-8") == "ok\n"


def test_reader_refuses_kill_and_ledger_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "no")
    claim_write_lock(tmp_path)
    assert not (tmp_path / ".dipcatcher-write.lock").exists()
    path = tmp_path / "kill_switch"
    with pytest.raises(OperatorKillError, match="write_owner_required"):
        write_operator_kill(path, "HALT_NEW_ORDERS")
    assert not path.exists()
    assert list(tmp_path.glob(".kill_switch.tmp")) == []
    with pytest.raises(WriteOwnerError, match="write_owner_required"):
        _atomic_write_text(tmp_path / "broker_state.json", "{}\n")
    with pytest.raises(WriteOwnerError, match="write_owner_required"):
        PaperLedger(tmp_path, "run-1")


def test_owner_must_claim_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "1")
    with pytest.raises(WriteOwnerError, match="write_owner_required"):
        require_write_owner()
    claim_write_lock(tmp_path)
    assert (tmp_path / ".dipcatcher-write.lock").is_file()
    _atomic_write_text(tmp_path / "broker_state.json", "{}\n")


def test_invalid_owner_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "maybe")
    with pytest.raises(WriteOwnerError, match="write_owner_invalid"):
        write_owner_mode()


def test_second_owner_fails_to_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holder = (
        "import os, time\n"
        "from pathlib import Path\n"
        "from quant_fund.utils.write_owner import claim_write_lock\n"
        "root = Path(os.environ['ROOT'])\n"
        "claim_write_lock(root)\n"
        "(root / 'ready').write_text('1', encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    env = os.environ.copy()
    env["ROOT"] = str(tmp_path)
    env["DIPCATCHER_WRITE_OWNER"] = "1"
    env.pop("DIPCATCHER_DATA_ROOT", None)
    process = subprocess.Popen(
        [sys.executable, "-c", holder],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(process.stderr.read() if process.stderr else "holder exited")
            time.sleep(0.05)
        monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "true")
        monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(tmp_path))
        with pytest.raises(WriteOwnerError, match="write_owner_busy"):
            claim_write_lock(tmp_path)
        from quant_fund.api.serve import _claim_write_owner

        with pytest.raises(SystemExit, match="DIPCATCHER_WRITE_OWNER"):
            _claim_write_owner()
    finally:
        process.kill()
        process.wait(timeout=5)


def test_sigusr1_drains_without_leaving_the_process() -> None:
    from quant_fund.api import serve

    previous = signal.getsignal(signal.SIGUSR1)
    reset_runtime_state()
    serve.install_drain_signal()
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert is_draining()
    finally:
        signal.signal(signal.SIGUSR1, previous)
        reset_runtime_state()


def test_proxy_contract_matches_the_app_cap() -> None:
    caddy = (_ROOT / "deploy" / "proxy" / "Caddyfile").read_text(encoding="utf-8")
    compose = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    makefile = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root_make = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert _MAX_REQUEST_BYTES == 64 * 1024
    assert caddy.count("max_size 64KiB") == 2
    assert "health_uri /ready" in caddy
    assert 'Strict-Transport-Security "max-age=31536000; includeSubDomains"' in caddy
    assert "{$QUANT_API_KEY}" in caddy
    assert "{$RESEARCH_API_KEY}" in caddy
    assert "BEGIN PRIVATE KEY" not in caddy
    assert "caddy:2.8.4" in compose
    assert 'command: ["dipcatcher", "smoke-up"]' in compose
    assert 'command: ["dipcatcher", "api"]' in compose
    assert 'DIPCATCHER_WRITE_OWNER: "1"' in compose
    assert 'DIPCATCHER_WRITE_OWNER: "0"' in compose
    assert "9464:9464" not in compose
    assert "0.0.0.0:" not in compose
    loaded = yaml.safe_load(compose)
    assert loaded["services"]["lab-a"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "1"
    assert loaded["services"]["lab-b"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"
    assert loaded["services"]["research-a"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"
    assert loaded["services"]["research-b"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"
    assert compose.count("/app/data:ro") == 1
    for name in (
        "prod-up",
        "prod-down",
        "prod-restart",
        "prod-smoke",
        "prod-preflight",
        "prod-backup",
        "prod-restore",
        "prod-dr-drill",
        "prod-rollback",
        "prod-secrets",
        "prod-canary",
        "prod-log-retain",
        "prod-backup-ship",
    ):
        assert f"{name}:" in makefile
        assert f"{name}:" not in root_make


def test_rolling_restart_zero_drop_contract() -> None:
    module = _load("dipcatcher_rolling_restart_ops7", _ROOT / "scripts" / "rolling_restart.py")
    assert module.drops_fail_the_run(["lab-a", "lab-b"], False)
    assert module.drops_fail_the_run(None, True)
    assert not module.drops_fail_the_run(None, False)
    assert module.load_has_transport({"status": {"transport": 2}})
    assert not module.load_has_transport({"status": {"200": 4}})
    assert module.load_has_transport({})
    args = module._parse(
        [
            "--containers",
            "lab-a",
            "lab-b",
            "--probe-url",
            "http://127.0.0.1:8000/ready",
        ]
    )
    assert args.signal == "USR1"
    with pytest.raises(SystemExit):
        module._parse([])


def test_https_load_verifies_the_dev_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = tmp_path / "dev.key"
    cert = tmp_path / "dev.crt"
    minted = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "1",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if minted.returncode != 0:
        pytest.skip("openssl could not mint a dev certificate")

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            body = b'{"status":"ok"}\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = _load("dipcatcher_http_load_ops7", _ROOT / "scripts" / "http_load.py")
    secret = "proxy-test-key-0123456789"
    monkeypatch.setenv("QUANT_API_KEY", secret)
    try:
        assert (
            script.main(
                [
                    "--url",
                    f"https://127.0.0.1:{port}/health",
                    "--concurrency",
                    "2",
                    "--seconds",
                    "1",
                    "--ca-file",
                    str(cert),
                    "--key-env",
                    "QUANT_API_KEY",
                ]
            )
            == 0
        )
    finally:
        server.shutdown()
        server.server_close()
    captured = capsys.readouterr().out
    assert secret not in captured
    report = json.loads(captured.strip().splitlines()[-1])
    assert report["http_5xx"] == 0
    assert report["status"].get("transport", 0) == 0
    assert report["ok"] > 0

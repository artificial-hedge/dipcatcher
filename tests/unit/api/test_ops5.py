"""Overload, crash-safe paper writes, and stderr-only service logs."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from quant_fund.api.service_config import ServiceConfigError, ServiceSettings
from quant_fund.utils.atomicio import atomic_write_text

_ROOT = Path(__file__).resolve().parents[3]
_PRIOR = '{"ok":1}\n'


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DIPCATCHER_RATE_LIMIT_RPS",
        "QUANT_API_KEY",
        "RESEARCH_API_KEY",
        "RESEARCH_API_HOST",
        "RESEARCH_API_PORT",
    ):
        monkeypatch.delenv(name, raising=False)


def _entrypoint():
    path = _ROOT / "docker" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("dipcatcher_entrypoint_ops5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script():
    path = _ROOT / "scripts" / "http_load.py"
    spec = importlib.util.spec_from_file_location("dipcatcher_http_load", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _killed(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_rate_limit_returns_429_and_health_stays_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app
    from quant_fund.api.rate_limit import reset_bucket

    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0.001")
    reset_bucket()
    with TestClient(app) as client:
        codes = [client.get("/ready").status_code for _ in range(30)]
        limited = client.get("/ready")
        health = client.get("/health")
    assert 429 in codes
    assert all(code < 500 for code in codes)
    assert limited.status_code == 429
    assert limited.json() == {"detail": "rate limit exceeded"}
    assert limited.headers["retry-after"]
    assert health.status_code == 200


def test_research_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.rate_limit import reset_bucket
    from quant_fund.api.research_api import app as research_app

    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0.001")
    reset_bucket()
    with TestClient(research_app) as client:
        codes = [client.get("/ready").status_code for _ in range(20)]
        assert client.get("/health").status_code == 200
    assert 429 in codes
    assert all(code < 500 for code in codes)


def test_invalid_rate_limit_is_429_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app
    from quant_fund.api.rate_limit import reset_bucket

    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "not-a-rate")
    reset_bucket()
    with TestClient(app) as client:
        denied = client.get("/ready")
        health = client.get("/health")
    assert denied.status_code == 429
    assert health.status_code == 200


def test_invalid_rate_limit_refuses_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("RESEARCH_API_HOST", "127.0.0.1")
    for raw in ("not-a-rate", "inf", "nan", "-1", "1000001"):
        monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", raw)
        with pytest.raises(ServiceConfigError, match="DIPCATCHER_RATE_LIMIT_RPS is invalid"):
            ServiceSettings.for_research()
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0")
    assert ServiceSettings.for_research().host == "127.0.0.1"
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "1.5")
    assert ServiceSettings.for_research().host == "127.0.0.1"


def test_full_data_dir_names_uid_10001(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entrypoint()
    target = tmp_path / "data"
    target.mkdir()

    def _full(*_args: object, **_kwargs: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(entry.os, "open", _full)
    message = entry.check_data_dir(str(target), require_write=True)
    assert message == f"data directory {target} is full; the dipcatcher user is uid 10001"
    assert "No space left on device" not in (message or "")


def test_sigkill_during_fsync_keeps_the_previous_ledger(tmp_path: Path) -> None:
    dest = tmp_path / "broker_state.json"
    dest.write_text(_PRIOR, encoding="utf-8")
    code = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "import quant_fund.utils.atomicio as atomicio\n"
        "def boom(_fd):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "atomicio.os.fsync = boom\n"
        "atomicio.atomic_write_text(Path(sys.argv[1]), '{\"ok\":2}\\n')\n"
    )
    proc = _killed(code, str(dest))
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    assert dest.read_text(encoding="utf-8") == _PRIOR
    assert json.loads(dest.read_text(encoding="utf-8")) == {"ok": 1}


def test_enospc_on_replace_keeps_the_previous_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "broker_state.json"
    atomic_write_text(dest, _PRIOR)

    def _full(_src: object, _dst: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", _full)
    with pytest.raises(OSError):
        atomic_write_text(dest, '{"ok":2}\n')
    assert dest.read_text(encoding="utf-8") == _PRIOR
    assert list(tmp_path.iterdir()) == [dest]


def test_readonly_directory_keeps_the_previous_ledger(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode bits")
    dest = tmp_path / "broker_state.json"
    atomic_write_text(dest, _PRIOR)
    os.chmod(tmp_path, 0o555)
    try:
        with pytest.raises(OSError):
            atomic_write_text(dest, '{"ok":2}\n')
        assert dest.read_text(encoding="utf-8") == _PRIOR
    finally:
        os.chmod(tmp_path, 0o755)


def test_service_logs_have_no_file_handler() -> None:
    from quant_fund.api.json_logging import json_log_config

    config = json_log_config("INFO")
    handler = config["handlers"]["stderr"]
    assert handler["class"] == "logging.StreamHandler"
    assert handler["stream"] == "ext://sys.stderr"
    assert "filename" not in handler
    for path in (_ROOT / "src" / "quant_fund" / "api").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "FileHandler" not in text
        assert "RotatingFileHandler" not in text


def test_http_load_records_latency_and_hides_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "planted-test-key-value-0123456789"

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            body = b'{"status":"ok"}\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env_file = tmp_path / "load.env"
    env_file.write_text(f"QUANT_API_KEY={secret}\n", encoding="utf-8")
    script = _load_script()
    previous = os.environ.get("QUANT_API_KEY")
    try:
        with pytest.raises(SystemExit):
            script.main(
                ["--url", "http://127.0.0.1/health", "--concurrency", "1", "--seconds", "121"]
            )
        assert (
            script.main(
                [
                    "--url",
                    f"http://127.0.0.1:{port}/health",
                    "--concurrency",
                    "2",
                    "--seconds",
                    "1",
                    "--env-file",
                    str(env_file),
                    "--key-env",
                    "QUANT_API_KEY",
                ]
            )
            == 0
        )
    finally:
        server.shutdown()
        server.server_close()
    assert os.environ.get("QUANT_API_KEY") == previous
    captured = capsys.readouterr().out
    assert secret not in captured
    report = json.loads(captured.strip().splitlines()[-1])
    assert report["http_5xx"] == 0
    assert report["ok"] > 0
    assert report["p50_ms"] is not None
    assert report["error_rate"] == 0.0

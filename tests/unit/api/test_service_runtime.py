"""Process contract: config, readiness, alerts, liveness probe, SIGTERM."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from quant_fund.api.alerts import plan_alert_actions
from quant_fund.api.json_logging import emit_service_event
from quant_fund.api.lifecycle import acceptance_block, begin_drain, reset_runtime_state
from quant_fund.api.service_config import ENV_CONTRACT, ServiceConfigError, ServiceSettings

_ROOT = Path(__file__).resolve().parents[3]
_PLANTED = "planted-test-key-value-0123456789"
_SERVICE_ENV = (
    "DIPCATCHER_ALERT_INTERVAL_S",
    "DIPCATCHER_BIND_HOST",
    "DIPCATCHER_BIND_PORT",
    "DIPCATCHER_CONFIG",
    "DIPCATCHER_DATA_ROOT",
    "DIPCATCHER_HALT_CANCELS_OPEN_ORDERS",
    "DIPCATCHER_KILL_DRAIN_DEBOUNCE_S",
    "DIPCATCHER_KILL_SWITCH_PATH",
    "DIPCATCHER_LOG_LEVEL",
    "DIPCATCHER_METRICS_HOST",
    "DIPCATCHER_METRICS_LOOPBACK_SCRAPE",
    "DIPCATCHER_METRICS_PORT",
    "DIPCATCHER_OBSERVE",
    "DIPCATCHER_OTEL_ENDPOINT",
    "DIPCATCHER_REQUIRE_DATA_WRITE",
    "DIPCATCHER_SHUTDOWN_GRACE_S",
    "DIPCATCHER_SLO_PATH",
    "PORT",
    "QUANT_API_KEY",
    "RESEARCH_API_HOST",
    "RESEARCH_API_KEY",
    "RESEARCH_API_PORT",
)


@pytest.fixture(autouse=True)
def _isolated_runtime() -> Any:
    reset_runtime_state()
    yield
    reset_runtime_state()


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SERVICE_ENV:
        monkeypatch.delenv(name, raising=False)


def _research(monkeypatch: pytest.MonkeyPatch) -> ServiceSettings:
    _clear(monkeypatch)
    monkeypatch.setenv("RESEARCH_API_HOST", "127.0.0.1")
    return ServiceSettings.for_research()


def test_discover_configs_dir_prefers_checkout_then_image(tmp_path: Path) -> None:
    from quant_fund.api.app import discover_configs_dir

    checkout = tmp_path / "checkout" / "src" / "quant_fund" / "api" / "app.py"
    checkout.parent.mkdir(parents=True)
    checkout.write_text("", encoding="utf-8")
    packaged = tmp_path / "checkout" / "configs"
    packaged.mkdir()
    image = tmp_path / "image" / "configs"
    image.mkdir(parents=True)
    assert discover_configs_dir(checkout, image) == packaged

    installed = (
        tmp_path / "image" / "lib" / "python" / "site-packages" / "quant_fund" / "api" / "app.py"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    assert discover_configs_dir(installed, image) == image


def test_env_contract_is_documented() -> None:
    runbook = (_ROOT / "docs" / "PRODUCTION_RUNBOOK.md").read_text(encoding="utf-8")
    example = (_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ENV_CONTRACT:
        assert name in runbook, name
        assert name in example, name


def test_container_contract() -> None:
    lab = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    research = (_ROOT / "docker" / "research-api.Dockerfile").read_text(encoding="utf-8")
    digest = "sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
    for text in (lab, research):
        assert digest in text
        assert "--uid 10001" in text
        assert "USER dipcatcher" in text
        assert "STOPSIGNAL SIGTERM" in text
        assert "HEALTHCHECK" in text
        assert "/app/healthcheck.py" in text
    assert 'CMD ["dipcatcher", "api"]' in lab
    assert "--require-hashes" in lab
    assert "docker/uv-requirements.txt" in lab
    assert "DIPCATCHER_BIND_PORT=8000" in lab
    assert "RESEARCH_API_PORT=8010" in research
    assert "--require-hashes" in research
    assert "docker/uv-requirements.txt" in research
    assert "DIPCATCHER_BIND_PORT" not in research


def test_loopback_research_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _research(monkeypatch)
    assert settings.host == "127.0.0.1"
    assert settings.port == 8010
    assert settings.log_level == "INFO"
    assert settings.shutdown_grace_s == 20
    assert settings.api_key_configured is False


def test_named_port_beats_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("RESEARCH_API_PORT", "9001")
    monkeypatch.setenv("PORT", "9002")
    assert ServiceSettings.for_research().port == 9001


def test_rejects_bad_port_without_echoing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("RESEARCH_API_PORT", _PLANTED)
    with pytest.raises(ServiceConfigError) as caught:
        ServiceSettings.for_research()
    assert "integer" in str(caught.value)
    assert _PLANTED not in str(caught.value)


def test_non_loopback_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("RESEARCH_API_HOST", "0.0.0.0")
    with pytest.raises(ServiceConfigError, match="requires RESEARCH_API_KEY"):
        ServiceSettings.for_research()
    monkeypatch.setenv("RESEARCH_API_KEY", "k")
    settings = ServiceSettings.for_research()
    assert settings.host == "0.0.0.0"
    assert settings.api_key_configured is True


def test_rejects_bad_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_LOG_LEVEL", "LOUD")
    with pytest.raises(ServiceConfigError, match="DIPCATCHER_LOG_LEVEL"):
        ServiceSettings.for_research()


def test_metrics_and_otel_require_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "9464")
    with pytest.raises(ServiceConfigError, match="DIPCATCHER_METRICS_PORT requires"):
        ServiceSettings.for_research()
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    with pytest.raises(ServiceConfigError, match="DIPCATCHER_OTEL_ENDPOINT requires"):
        ServiceSettings.for_research()


def test_otel_userinfo_is_rejected_and_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv(
        "DIPCATCHER_OTEL_ENDPOINT",
        f"http://user:{_PLANTED}@127.0.0.1:4318/v1/traces",
    )
    with pytest.raises(ServiceConfigError) as caught:
        ServiceSettings.for_research()
    assert "must not embed credentials" in str(caught.value)
    assert _PLANTED not in str(caught.value)


def test_explicit_slo_must_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "slo.yml"
    path.write_text("version: 9\n", encoding="utf-8")
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_SLO_PATH", str(path))
    with pytest.raises(ServiceConfigError, match="requires DIPCATCHER_OBSERVE"):
        ServiceSettings.for_research()
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    with pytest.raises(ServiceConfigError, match="failed validation"):
        ServiceSettings.for_research()


def test_observe_discovers_shipped_slo(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    settings = ServiceSettings.for_research()
    assert settings.observe is True
    assert settings.slo_path is not None
    assert settings.slo_path.endswith("deploy/observability/slo.yml")


def test_kill_switch_path_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-dir" / "switch"
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_KILL_SWITCH_PATH", str(missing))
    with pytest.raises(ServiceConfigError, match="directory does not exist"):
        ServiceSettings.for_research()

    target = tmp_path / "target"
    target.write_text("ENABLED\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.setenv("DIPCATCHER_KILL_SWITCH_PATH", str(link))
    with pytest.raises(ServiceConfigError, match="symlink"):
        ServiceSettings.for_research()

    bad = tmp_path / "bad"
    bad.write_text(_PLANTED + "\n", encoding="utf-8")
    monkeypatch.setenv("DIPCATCHER_KILL_SWITCH_PATH", str(bad))
    with pytest.raises(ServiceConfigError) as caught:
        ServiceSettings.for_research()
    assert "kill_switch_invalid" in str(caught.value)
    assert _PLANTED not in str(caught.value)


def test_lab_non_loopback_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(ServiceConfigError, match="requires QUANT_API_KEY"):
        ServiceSettings.for_lab("0.0.0.0", 8000)


def test_service_event_redacts_secret_values(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUANT_API_KEY", _PLANTED)
    emit_service_event("service_start", note=f"prefix {_PLANTED} suffix")
    err = capsys.readouterr().err
    assert _PLANTED not in err
    assert "[redacted]" in err
    payload = json.loads(err.strip())
    assert payload["event"] == "service_start"


def test_plan_ignores_no_data_and_edges_fail() -> None:
    assert plan_alert_actions({}, {"stage_latency_p99": "no_data"}) == []
    assert plan_alert_actions({"stage_latency_p99": "fail"}, {"stage_latency_p99": "no_data"}) == []
    assert plan_alert_actions({}, {"stage_latency_p99": "fail"}) == [("stage_latency_p99", "alarm")]
    previous = {"stage_latency_p99": "fail"}
    assert plan_alert_actions(previous, {"stage_latency_p99": "fail"}) == []
    assert plan_alert_actions(previous, {"stage_latency_p99": "pass"}) == [
        ("stage_latency_p99", "resolve")
    ]


def test_alert_monitor_alarms_once_then_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from quant_fund.api.alerts import AlertMonitor
    from quant_fund.observe.logging import set_log_sink

    switch = tmp_path / "switch"
    switch.write_text("ENABLED\n", encoding="utf-8")
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("DIPCATCHER_KILL_SWITCH_PATH", str(switch))
    snapshot = {"latency_p99_seconds": 3.0, "error_ratio": 0.0, "freshness_seconds": 1.0}
    monkeypatch.setattr("quant_fund.api.alerts.current_slo_snapshot", lambda: dict(snapshot))
    records: list[dict[str, Any]] = []
    set_log_sink(records)
    try:
        monitor = AlertMonitor(
            _MonitorSettings(str(_ROOT / "deploy" / "observability" / "slo.yml"))
        )
        first = monitor.tick()
        assert ("stage_latency_p99", "alarm") in first
        assert ("kill_switch", "alarm") not in first
        assert monitor.tick() == []
        snapshot["latency_p99_seconds"] = 0.001
        second = monitor.tick()
        assert ("stage_latency_p99", "resolve") in second
        switch.write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
        halted = monitor.tick()
        assert ("kill_switch", "alarm") in halted
        switch.write_text("ENABLED\n", encoding="utf-8")
        assert ("kill_switch", "resolve") in monitor.tick()
        begin_drain()
        drained = monitor.tick()
        assert all(rule != "kill_switch" for rule, _action in drained)
    finally:
        set_log_sink(None)
    actions = [row["action"] for row in records if row["event"] == "alert"]
    assert actions.count("alarm") >= 2
    assert "resolve" in actions


class _MonitorSettings:
    service = "lab"
    observe = True
    alert_interval_s = 60

    def __init__(self, slo_path: str) -> None:
        self.slo_path: str | None = slo_path

    def public_dict(self) -> dict[str, str | int | bool | None]:
        return {"service": self.service}


def test_health_stays_up_while_ready_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app

    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.delenv("DIPCATCHER_KILL_SWITCH_PATH", raising=False)
    client = TestClient(app)
    begin_drain()
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["report"]["accepting_work"] == "draining"
    assert client.get("/health").status_code == 200


def test_kill_switch_file_blocks_readiness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app

    switch = tmp_path / "switch"
    switch.write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.setenv("DIPCATCHER_KILL_SWITCH_PATH", str(switch))
    ready = TestClient(app).get("/ready")
    assert ready.status_code == 503
    body = ready.json()
    assert body["report"]["accepting_work"] == "kill_switch:HALT_NEW_ORDERS"
    assert "HALT_NEW_ORDERS" not in json.dumps(body["checks"])


def test_research_ready_and_remote_refusal(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.research_api import ResearchApiSettings, create_app

    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
    )
    app = create_app(settings)
    local = TestClient(app)
    missing = local.get("/ready")
    assert missing.status_code == 503
    assert missing.json()["checks"]["data_root"] is False
    for directory in (
        settings.data_root,
        settings.receipts_dir,
        settings.verifier_dir,
        settings.artifacts_dir,
    ):
        directory.mkdir()
    assert local.get("/ready").status_code == 200
    begin_drain()
    drained = local.get("/ready")
    assert drained.status_code == 503
    assert drained.json()["report"]["accepting_work"] == "draining"
    remote = TestClient(create_app(settings), client=("203.0.113.7", 9))
    assert remote.get("/health").status_code == 200
    assert remote.get("/ready").status_code == 403


def test_sigterm_handler_drains_before_server_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn
    from fastapi import FastAPI

    from quant_fund.api.serve import _DrainServer

    seen: dict[str, int] = {}

    def record(self: object, sig: int, frame: object) -> None:
        seen["sig"] = sig

    monkeypatch.setattr(uvicorn.Server, "handle_exit", record)
    config = uvicorn.Config(FastAPI(), host="127.0.0.1", port=9, log_config=None)
    _DrainServer(config).handle_exit(signal.SIGTERM, None)
    assert seen["sig"] == signal.SIGTERM
    assert acceptance_block() == "draining"


def _healthcheck():
    path = _ROOT / "docker" / "healthcheck.py"
    spec = importlib.util.spec_from_file_location("dipcatcher_healthcheck", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_healthcheck_probes_local_health(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _clear(monkeypatch)
        monkeypatch.setenv("DIPCATCHER_BIND_PORT", str(port))
        assert _healthcheck().main() == 0
        monkeypatch.setenv("DIPCATCHER_BIND_PORT", "not-a-port")
        assert _healthcheck().main() == 1
    finally:
        server.shutdown()
        server.server_close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _child_env(port: int) -> dict[str, str]:
    keep = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(
        {
            "PORT": str(port),
            "DIPCATCHER_LOG_LEVEL": "INFO",
            "PYTHONDONTWRITEBYTECODE": "1",
            "QUANT_API_KEY": _PLANTED,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "KMP_INIT_AT_FORK": "FALSE",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _redact(text: str) -> str:
    return text.replace(_PLANTED, "[redacted]")


def test_sigterm_shuts_down_the_service() -> None:
    port = _free_port()
    script = textwrap.dedent(
        """
        import sys
        from fastapi import FastAPI
        from quant_fund.api.lifecycle import service_lifespan
        from quant_fund.api.serve import _serve
        from quant_fund.api.service_config import ServiceSettings

        app = FastAPI(lifespan=service_lifespan)

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        settings = ServiceSettings(
            service="lab",
            host="127.0.0.1",
            port=int(sys.argv[1]),
            log_level="INFO",
            config_path=None,
            observe=False,
            metrics_host="127.0.0.1",
            metrics_port=None,
            otel_configured=False,
            shutdown_grace_s=1,
            kill_switch_path=None,
            slo_path=None,
            alert_interval_s=60,
            api_key_configured=True,
        )
        _serve(settings, app)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(port)],
        env=_child_env(port),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 90
        last_error = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if int(response.status) == 200:
                        assert response.headers.get("Server") is None
                        break
            except (OSError, URLError, TimeoutError) as exc:
                last_error = str(exc)
            else:
                continue
            time.sleep(0.2)
        else:
            raise AssertionError(_redact(f"service did not become healthy: {last_error}"))
        assert proc.poll() is None, _redact(proc.stderr.read() if proc.stderr else "")
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    stdout, stderr = proc.communicate()
    combined = _redact(stdout + stderr)
    assert code in {0, -signal.SIGTERM}, combined
    assert _PLANTED not in (stdout + stderr)
    assert "service_draining" in stderr
    assert "service_shutdown" in stderr
    assert stderr.index("service_draining") < stderr.index("service_shutdown")

"""Metrics auth, the shared kill file, and the SYNTHETIC readiness bootstrap."""

from __future__ import annotations

import http.client
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from quant_fund.api.lifecycle import acceptance_block, begin_drain, reset_runtime_state
from quant_fund.api.service_config import ServiceConfigError, ServiceSettings
from quant_fund.config.models import KillSwitchConfig
from quant_fund.execution.order_recon import OrderFillLedger
from quant_fund.monitoring.kill_switch import VALID_STATES, KillSwitch
from quant_fund.observe.metrics import set_data_freshness
from quant_fund.observe.serve import start_metrics_server, stop_metrics_server
from quant_fund.schemas.errors import KillSwitchActive
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus
from quant_fund.utils.operator_kill import (
    ENV_PATH,
    STATES,
    OperatorKillError,
    latch_drain,
    orders_blocked,
    write_operator_kill,
)

_ROOT = Path(__file__).resolve().parents[3]
_KEY = "metrics-test-key-0123456789"
_SERVICE_ENV = (
    "DIPCATCHER_BIND_HOST",
    "DIPCATCHER_CONFIG",
    "DIPCATCHER_KILL_SWITCH_PATH",
    "DIPCATCHER_METRICS_HOST",
    "DIPCATCHER_METRICS_LOOPBACK_SCRAPE",
    "DIPCATCHER_METRICS_PORT",
    "DIPCATCHER_OBSERVE",
    "DIPCATCHER_OTEL_ENDPOINT",
    "DIPCATCHER_SLO_PATH",
    "PORT",
    "QUANT_API_KEY",
    "RESEARCH_API_HOST",
    "RESEARCH_API_KEY",
)


@pytest.fixture(autouse=True)
def _reset_runtime() -> Iterator[None]:
    reset_runtime_state()
    stop_metrics_server()
    yield
    reset_runtime_state()
    stop_metrics_server()


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SERVICE_ENV:
        monkeypatch.delenv(name, raising=False)


def _config(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    path = tmp_path / "research.yaml"
    path.write_text(
        f"data:\n  root: {root.as_posix()!r}\n  source: synthetic\nruntime:\n  mode: research\n",
        encoding="utf-8",
    )
    return path


def _order() -> Order:
    stamp = datetime(2026, 11, 2, tzinfo=UTC)
    return Order(
        order_id="ops2-order",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=stamp,
        decision_time=stamp,
        order_time=stamp,
        status=OrderStatus.NEW,
    )


@contextmanager
def _metrics(*, allow_unauthenticated: bool = False, key_env: str = "") -> Iterator[int]:
    port = start_metrics_server(0, allow_unauthenticated=allow_unauthenticated, key_env=key_env)
    try:
        yield port
    finally:
        stop_metrics_server()


def _scrape(port: int, headers: dict[str, str] | None = None) -> tuple[int, bytes, str | None]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/metrics", headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        return response.status, body, response.getheader("WWW-Authenticate")
    finally:
        connection.close()


def test_metrics_requires_the_same_key_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    set_data_freshness(3.5)
    with _metrics(key_env="QUANT_API_KEY") as port:
        status, body, challenge = _scrape(port)
        assert status == 401
        assert body == b'{"detail":"invalid or missing X-API-Key"}\n'
        assert challenge == "Bearer"
        status, body, _challenge = _scrape(port, {"X-API-Key": _KEY})
        assert status == 200
        assert b"dipcatcher_" in body
        status, _body, _challenge = _scrape(port, {"Authorization": f"Bearer {_KEY}"})
        assert status == 200
        status, _body, _challenge = _scrape(port, {"Authorization": f"BEARER {_KEY}"})
        assert status == 200
        status, _body, _challenge = _scrape(
            port,
            {"X-API-Key": _KEY, "Authorization": "Bearer other-test-key-0123456789"},
        )
        assert status == 401
        status, _body, _challenge = _scrape(port, {"Authorization": "Bearer"})
        assert status == 401
        status, _body, _challenge = _scrape(port, {"X-API-Key": "wrong-test-key-0123456789"})
        assert status == 401


def test_metrics_loopback_opt_out_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    set_data_freshness(3.5)
    with _metrics(allow_unauthenticated=True) as port:
        status, body, challenge = _scrape(port)
        assert status == 200
        assert b"dipcatcher_" in body
        assert challenge is None
    with pytest.raises(ValueError, match="loopback-only"):
        start_metrics_server(0, host="0.0.0.0", allow_unauthenticated=True)


def test_service_settings_fail_closed_on_metrics_exposure(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "9464")
    with pytest.raises(ServiceConfigError, match="DIPCATCHER_METRICS_LOOPBACK_SCRAPE=1"):
        ServiceSettings.for_lab("127.0.0.1", 8000)

    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("DIPCATCHER_METRICS_HOST", "0.0.0.0")
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "9464")
    monkeypatch.setenv("DIPCATCHER_METRICS_LOOPBACK_SCRAPE", "1")
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    with pytest.raises(ServiceConfigError, match="loopback DIPCATCHER_METRICS_HOST"):
        ServiceSettings.for_lab("127.0.0.1", 8000)

    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_METRICS_LOOPBACK_SCRAPE", "1")
    with pytest.raises(ServiceConfigError, match="requires DIPCATCHER_METRICS_PORT"):
        ServiceSettings.for_research()

    _clear(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "9464")
    monkeypatch.setenv("DIPCATCHER_METRICS_LOOPBACK_SCRAPE", "1")
    settings = ServiceSettings.for_lab("127.0.0.1", 8000)
    assert settings.metrics_loopback_scrape is True
    assert settings.metrics_host == "127.0.0.1"


def test_ready_accepts_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app

    _clear(monkeypatch)
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    with TestClient(app) as client:
        denied = client.get("/ready")
        assert denied.status_code == 401
        assert denied.json() == {"detail": "invalid or missing X-API-Key"}
        allowed = client.get("/ready", headers={"Authorization": f"Bearer {_KEY}"})
        assert allowed.status_code in {200, 503}
        assert allowed.json()["status"] in {"ready", "not_ready"}


def test_kill_states_match_and_fail_closed(tmp_path: Path) -> None:
    assert STATES == VALID_STATES
    missing = tmp_path / "absent"
    assert orders_blocked(missing) is None
    enabled = tmp_path / "enabled"
    enabled.write_text("ENABLED\n", encoding="utf-8")
    assert orders_blocked(enabled) is None
    halted = tmp_path / "halted"
    halted.write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
    assert orders_blocked(halted) == "kill_switch:HALT_NEW_ORDERS"
    unknown = tmp_path / "unknown"
    unknown.write_text("NOT_A_STATE\n", encoding="utf-8")
    assert orders_blocked(unknown) == "kill_switch_invalid"
    huge = tmp_path / "huge"
    huge.write_text("x" * 65, encoding="utf-8")
    assert orders_blocked(huge) == "kill_switch_invalid"
    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff\xfe")
    assert orders_blocked(binary) == "kill_switch_invalid"
    directory = tmp_path / "directory"
    directory.mkdir()
    assert orders_blocked(directory) == "kill_switch_directory"
    target = tmp_path / "target"
    target.write_text("ENABLED\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert orders_blocked(link) == "kill_switch_symlink"
    with pytest.raises(OperatorKillError, match="kill_switch_symlink"):
        write_operator_kill(link, "HALT_NEW_ORDERS")
    assert target.read_text(encoding="utf-8").strip() == "ENABLED"
    with pytest.raises(OperatorKillError, match="invalid kill switch state"):
        write_operator_kill(tmp_path / "fresh", "NOT_A_STATE")
    assert not (tmp_path / "fresh").exists()


def test_paper_orders_follow_the_file_without_mutating_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "kill_switch"
    path.write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PATH, str(path))
    switch = KillSwitch(KillSwitchConfig())
    with pytest.raises(KillSwitchActive, match="kill_switch:HALT_NEW_ORDERS"):
        switch.assert_new_orders_allowed()
    assert switch.state == "ENABLED"
    ledger = OrderFillLedger()
    with pytest.raises(KillSwitchActive, match="kill_switch:HALT_NEW_ORDERS"):
        ledger.submit(_order())
    assert ledger.kill_state == "ENABLED"
    path.write_text("NOT_A_STATE\n", encoding="utf-8")
    with pytest.raises(KillSwitchActive, match="kill_switch_invalid"):
        switch.assert_new_orders_allowed()
    path.write_text("ENABLED\n", encoding="utf-8")
    switch.assert_new_orders_allowed()
    assert ledger.submit(_order()) == "accepted"


def test_acceptance_block_rereads_and_drain_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "kill_switch"
    path.write_text("CANCEL_OPEN_ORDERS\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PATH, str(path))
    assert acceptance_block() == "kill_switch:CANCEL_OPEN_ORDERS"
    begin_drain()
    assert acceptance_block() == "draining"
    reset_runtime_state()
    path.write_text("ENABLED\n", encoding="utf-8")
    assert acceptance_block() is None


def test_latch_drain_is_one_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "kill_switch"
    path.write_text("ENABLED\n", encoding="utf-8")
    event = threading.Event()
    assert latch_drain(event, path) is False
    assert not event.is_set()
    path.write_text("FLATTEN_OPTIONAL\n", encoding="utf-8")
    assert latch_drain(event, path) is True
    path.write_text("ENABLED\n", encoding="utf-8")
    assert latch_drain(event, path) is False
    assert event.is_set()
    directory = tmp_path / "blocked"
    directory.mkdir()
    other = threading.Event()
    assert latch_drain(other, directory) is True
    monkeypatch.setenv(ENV_PATH, str(path))
    path.write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
    from_env = threading.Event()
    assert latch_drain(from_env) is True


def test_harness_serve_wrapper_latches_from_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    path = tmp_path / "kill_switch"
    path.write_text("ENABLED\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PATH, str(path))
    event = threading.Event()
    app = SimpleNamespace(state=SimpleNamespace(metrics=SimpleNamespace(draining=event)))

    def fake_run(*_args: object, **_kwargs: object) -> None:
        path.write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
        assert event.wait(2)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    from fx1.serve.kill_drain import serve_until_signal

    serve_until_signal(app, host="127.0.0.1", port=9)
    assert event.is_set()


def test_live_kill_writes_before_the_desk_and_reset_writes_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from quant_fund.cli.live_cmds import live_kill, live_reset
    from quant_fund.utils.operator_kill import publish_operator_kill

    path = tmp_path / "kill_switch"
    calls: list[str] = []

    def desk(*_args: object, **_kwargs: object) -> None:
        calls.append("desk")
        if path.is_file():
            assert path.read_text(encoding="utf-8").strip() == "HALT_NEW_ORDERS"

    monkeypatch.setattr("quant_fund.cli.live_cmds._use_gateway", lambda *_a, **_k: False)
    monkeypatch.setattr("quant_fund.cli.live_cmds._desk", desk)
    live_kill(
        reason="ops-test",
        actor="",
        root=tmp_path,
        config=None,
        ts_ns=1,
        at=None,
        state_dir=None,
        kill_switch_file=path,
    )
    assert calls == ["desk"]
    assert path.read_text(encoding="utf-8").strip() == "HALT_NEW_ORDERS"

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise OperatorKillError("kill_switch_directory")

    monkeypatch.setattr("quant_fund.utils.operator_kill.publish_operator_kill", refuse)
    with pytest.raises(OperatorKillError, match="kill_switch_directory"):
        live_kill(
            reason="ops-test",
            actor="",
            root=tmp_path,
            config=None,
            ts_ns=1,
            at=None,
            state_dir=None,
            kill_switch_file=path,
        )
    assert calls == ["desk"]

    def broken_desk(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("desk down")

    monkeypatch.setattr("quant_fund.cli.live_cmds._desk", broken_desk)
    with pytest.raises(RuntimeError, match="desk down"):
        live_reset(
            reason="ops-test",
            actor="",
            root=tmp_path,
            config=None,
            ts_ns=1,
            at=None,
            state_dir=None,
            kill_switch_file=path,
        )
    assert path.read_text(encoding="utf-8").strip() == "HALT_NEW_ORDERS"

    monkeypatch.setattr("quant_fund.cli.live_cmds._desk", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "quant_fund.utils.operator_kill.publish_operator_kill",
        publish_operator_kill,
    )
    live_reset(
        reason="ops-test",
        actor="",
        root=tmp_path,
        config=None,
        ts_ns=1,
        at=None,
        state_dir=None,
        kill_switch_file=path,
    )
    assert path.read_text(encoding="utf-8").strip() == "ENABLED"


def test_paper_halt_writes_only_the_configured_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from quant_fund.cli.ops_cmds import _publish_paper_halt

    path = tmp_path / "kill_switch"
    monkeypatch.delenv(ENV_PATH, raising=False)
    _publish_paper_halt("HALT_NEW_ORDERS")
    assert not path.exists()
    monkeypatch.setenv(ENV_PATH, str(path))
    _publish_paper_halt("HALT_NEW_ORDERS")
    assert path.read_text(encoding="utf-8").strip() == "HALT_NEW_ORDERS"
    _publish_paper_halt("ENABLED")
    assert path.read_text(encoding="utf-8").strip() == "ENABLED"
    monkeypatch.setenv(ENV_PATH, str(tmp_path / "missing" / "kill_switch"))
    with pytest.raises(typer.BadParameter, match="kill_switch_directory"):
        _publish_paper_halt("HALT_NEW_ORDERS")


def test_ready_bootstrap_makes_doctor_ready_and_refuses_to_clobber(tmp_path: Path) -> None:
    from quant_fund.pipeline.readiness_bootstrap import (
        ReadinessBootstrapError,
        bootstrap_readiness,
    )

    config = _config(tmp_path)
    report = bootstrap_readiness(config)
    assert report["data_manifest"] == "ok"
    assert report["research_receipt"] == "ok"
    assert report.get("live") != "REJECTED"
    receipt = tmp_path / "data" / "metadata" / "research" / "latest.json"
    original = receipt.read_text(encoding="utf-8")
    assert "SYNTHETIC" in original
    again = bootstrap_readiness(config)
    assert again["data_manifest"] == "ok"
    assert receipt.read_text(encoding="utf-8") == original

    broken = tmp_path / "broken"
    broken_config = broken / "research.yaml"
    broken.mkdir()
    data_root = broken / "data"
    broken_config.write_text(
        f"data:\n  root: {data_root.as_posix()!r}\n  source: synthetic\n",
        encoding="utf-8",
    )
    manifest = data_root / "metadata" / "data_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(ReadinessBootstrapError, match="left unchanged"):
        bootstrap_readiness(broken_config)
    assert manifest.read_text(encoding="utf-8") == "{"

    linked = tmp_path / "linked"
    linked_config = linked / "research.yaml"
    linked.mkdir()
    linked_root = linked / "data"
    (linked_root / "metadata").mkdir(parents=True)
    (linked_root / "metadata" / "data_manifest.json").symlink_to(manifest)
    linked_config.write_text(
        f"data:\n  root: {linked_root.as_posix()!r}\n  source: synthetic\n",
        encoding="utf-8",
    )
    with pytest.raises(ReadinessBootstrapError, match="symlink"):
        bootstrap_readiness(linked_config)


def test_ready_bootstrap_command(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    result = CliRunner().invoke(cli, ["ready-bootstrap", "--config", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "data_manifest=ok" in result.output
    assert "research_receipt=ok" in result.output
    assert "synthetic=true" in result.output


def test_uv_installer_is_hash_pinned() -> None:
    text = (_ROOT / "docker" / "uv-requirements.txt").read_text(encoding="utf-8")
    pinned = "7a85330de0a7eb0d5c6cf03c80edfb86facad19df367a0b52fc906db1ab15ce9"
    assert text.startswith("uv==0.11.23")
    assert pinned in text
    assert text.count("--hash=sha256:") == 19
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    research = (_ROOT / "docker" / "research-api.Dockerfile").read_text(encoding="utf-8")
    for image in (dockerfile, research):
        assert "--require-hashes" in image
        assert "docker/uv-requirements.txt" in image
    compose = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    assert 'command: ["dipcatcher", "smoke-up"]' in compose
    assert "DIPCATCHER_METRICS_LOOPBACK_SCRAPE" in compose

"""Module-entry kill watch, operator gauges, and the alert pack."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from quant_fund.utils.operator_kill import ENV_PATH, write_operator_kill

_ROOT = Path(__file__).resolve().parents[3]
_OBS = _ROOT / "deploy" / "observability"

_FAKE_UVICORN = """\
import os
import threading
import time
from pathlib import Path

def run(app, host="127.0.0.1", port=8000, **kwargs):
    names = [item.name for item in threading.enumerate()]
    if "fx1-operator-kill" not in names:
        raise SystemExit("missing fx1-operator-kill")
    if kwargs.get("reload") is not False or kwargs.get("server_header") is not False:
        raise SystemExit("pinned kwargs missing")
    if str(host) != "127.0.0.1" or int(port) != 9:
        raise SystemExit("bad bind")
    path = Path(os.environ["DIPCATCHER_KILL_SWITCH_PATH"])
    path.write_text("HALT_NEW_ORDERS\\n", encoding="utf-8")
    if not app.state.metrics.draining.wait(2):
        raise SystemExit("drain not latched")
    path.write_text("ENABLED\\n", encoding="utf-8")
    deadline = time.monotonic() + 2
    while app.state.metrics.draining.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)
    if app.state.metrics.draining.is_set():
        raise SystemExit("drain not cleared")
    print("WATCH_OK", flush=True)
"""


def test_plain_import_does_not_install_the_module_hook() -> None:
    from fx1.serve.module_entry import _ApiEntryFinder

    assert not any(isinstance(finder, _ApiEntryFinder) for finder in sys.meta_path)


def test_arm_kill_watch_calls_the_saved_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    from fx1.serve.module_entry import arm_kill_watch

    path = tmp_path / "kill_switch"
    write_operator_kill(path, "HALT_NEW_ORDERS")
    monkeypatch.setenv(ENV_PATH, str(path))
    monkeypatch.setenv("DIPCATCHER_KILL_DRAIN_DEBOUNCE_S", "0")
    calls: list[tuple[object, object]] = []
    saved = uvicorn.run

    def fake(app: object, host: str = "127.0.0.1", port: int = 8000, **kwargs: object) -> None:
        calls.append((host, port))
        assert kwargs.get("reload") is False
        assert kwargs.get("server_header") is False
        assert app.state.metrics.draining.is_set()

    try:
        uvicorn.run = fake  # type: ignore[method-assign]
        arm_kill_watch()
        wrapped = uvicorn.run
        assert getattr(wrapped, "__dipcatcher_kill_watch__", False)
        arm_kill_watch()
        assert uvicorn.run is wrapped
        event = threading.Event()
        app = SimpleNamespace(state=SimpleNamespace(metrics=SimpleNamespace(draining=event)))
        uvicorn.run(app, host="127.0.0.1", port=9)
    finally:
        uvicorn.run = saved  # type: ignore[method-assign]
    assert calls == [("127.0.0.1", 9)]
    assert event.is_set()


def test_module_entrypoint_watches_the_kill_file(tmp_path: Path) -> None:
    fake_dir = tmp_path / "fake_uvicorn"
    fake_dir.mkdir()
    (fake_dir / "uvicorn.py").write_text(_FAKE_UVICORN, encoding="utf-8")
    kill = tmp_path / "kill_switch"
    write_operator_kill(kill, "ENABLED")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(fake_dir), env.get("PYTHONPATH", "")])
    env[ENV_PATH] = str(kill)
    env["DIPCATCHER_KILL_DRAIN_DEBOUNCE_S"] = "0"
    env["FX1_API_HOST"] = "127.0.0.1"
    env["FX1_API_PORT"] = "9"
    env.pop("FX1_API_KEY", None)
    proc = subprocess.run(
        [sys.executable, "-m", "fx1.serve.api"],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:] + proc.stdout[-1000:]
    assert "WATCH_OK" in proc.stdout
    assert "operator_kill_drain_latched" in proc.stderr
    assert "kill_switch:HALT_NEW_ORDERS" in proc.stderr
    assert "operator_kill_drain_cleared" in proc.stderr


def test_operator_gauges_follow_readiness_and_the_kill_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.api.lifecycle import begin_drain, reset_runtime_state
    from quant_fund.api.ops_metrics import install_ops_sampler
    from quant_fund.observe.metrics import clear_ops_sampler, render_prometheus

    path = tmp_path / "kill_switch"
    write_operator_kill(path, "ENABLED")
    monkeypatch.setenv(ENV_PATH, str(path))
    reset_runtime_state()
    install_ops_sampler()
    try:
        text = render_prometheus()
        assert "dipcatcher_ready 1\n" in text
        assert "dipcatcher_draining 0\n" in text
        assert "dipcatcher_kill_switch_blocking 0\n" in text
        assert 'dipcatcher_kill_switch_info{code="enabled"} 1\n' in text
        write_operator_kill(path, "HALT_NEW_ORDERS")
        halted = render_prometheus()
        assert "dipcatcher_ready 0\n" in halted
        assert "dipcatcher_kill_switch_blocking 1\n" in halted
        assert 'dipcatcher_kill_switch_info{code="kill_switch:HALT_NEW_ORDERS"} 1\n' in halted
        path.write_text("nope\n", encoding="utf-8")
        invalid = render_prometheus()
        assert "nope" not in invalid
        assert 'code="kill_switch_invalid"' in invalid
        begin_drain()
        write_operator_kill(path, "ENABLED")
        drained = render_prometheus()
        assert "dipcatcher_ready 0\n" in drained
        assert "dipcatcher_draining 1\n" in drained
        assert "dipcatcher_kill_switch_blocking 0\n" in drained
    finally:
        clear_ops_sampler()
        reset_runtime_state()


def test_render_omits_operator_gauges_without_a_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.observe.metrics import clear_ops_sampler, render_prometheus

    path = tmp_path / "kill_switch"
    write_operator_kill(path, "FLATTEN")
    monkeypatch.setenv(ENV_PATH, str(path))
    clear_ops_sampler()
    assert "dipcatcher_ready" not in render_prometheus()
    assert "FLATTEN" not in render_prometheus()


def test_unsafe_sampler_values_are_not_exposed() -> None:
    from quant_fund.observe.metrics import (
        clear_ops_sampler,
        register_ops_sampler,
        render_prometheus,
    )

    register_ops_sampler(lambda: (2.0, 0.0, 0.0, "enabled"))
    try:
        assert "dipcatcher_ready" not in render_prometheus()
    finally:
        clear_ops_sampler()
    register_ops_sampler(lambda: (1.0, 0.0, 0.0, 'ena"bled'))
    try:
        text = render_prometheus()
    finally:
        clear_ops_sampler()
    assert 'ena"bled' not in text
    assert 'code="kill_switch_invalid"' in text


def test_alert_pack_covers_slo_readiness_drain_kill_and_errors() -> None:
    rules_path = _OBS / "alerts.yml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    expressions = [rule["expr"] for rule in rules["groups"][0]["rules"]]
    blob = "\n".join(expressions)
    for name in (
        "dipcatcher_ready",
        "dipcatcher_draining",
        "dipcatcher_kill_switch_blocking",
        "dipcatcher_stage_latency_seconds_bucket",
        "dipcatcher_errors_total",
        "dipcatcher_stage_latency_seconds_count",
        "dipcatcher_data_freshness_seconds",
    ):
        assert name in blob
    assert "> 2" in blob
    assert ">= 0.01" in blob
    assert ">= 86400" in blob
    assert "otel_export" in blob
    dashboard = json.loads((_OBS / "grafana/dashboards/dipcatcher-ops.json").read_text())
    rendered = json.dumps(dashboard)
    assert dashboard["uid"] == "dipcatcher-ops"
    assert dashboard["schemaVersion"] == 39
    assert "not a live P&L claim" in dashboard["description"]
    assert "research scores" in dashboard["description"]
    for name in (
        "dipcatcher_ready",
        "dipcatcher_draining",
        "dipcatcher_kill_switch_blocking",
        "dipcatcher_kill_switch_info",
        "dipcatcher_stage_latency_seconds_bucket",
        "dipcatcher_errors_total",
        "dipcatcher_data_freshness_seconds",
    ):
        assert name in rendered
    assert "sharpe" not in rendered.lower()
    assert "sortino" not in rendered.lower()
    prometheus = yaml.safe_load((_OBS / "prometheus.yml").read_text(encoding="utf-8"))
    assert prometheus["rule_files"] == ["alerts.yml", "slo_rules.yml"]
    compose = (_OBS / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./alerts.yml:/etc/prometheus/alerts.yml:ro" in compose
    assert "./slo_rules.yml:/etc/prometheus/slo_rules.yml:ro" in compose
    runbook = (_ROOT / "docs" / "PRODUCTION_RUNBOOK.md").read_text(encoding="utf-8")
    assert "python -m fx1.serve.api" in runbook
    assert "write_operator_kill" in runbook
    assert "paper_state_backup.py restore" in runbook
    assert "dipcatcher:previous" in runbook
    assert "Incident response" in runbook

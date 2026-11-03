"""Production profile, compose scrape contract, and rolling restart."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

from quant_fund.api.service_config import ServiceConfigError, ServiceSettings

_ROOT = Path(__file__).resolve().parents[3]
_CLEAR = (
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
    "DIPCATCHER_PROFILE",
    "DIPCATCHER_WRITE_OWNER",
    "DIPCATCHER_RATE_LIMIT_RPS",
    "DIPCATCHER_REQUIRE_DATA_WRITE",
    "DIPCATCHER_SHUTDOWN_GRACE_S",
    "DIPCATCHER_SLO_PATH",
    "PORT",
    "QUANT_API_KEY",
    "RESEARCH_API_HOST",
    "RESEARCH_API_KEY",
    "RESEARCH_API_PORT",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CLEAR:
        monkeypatch.delenv(name, raising=False)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entrypoint():
    return _load(_ROOT / "docker" / "entrypoint.py", "dipcatcher_entrypoint_ops6")


def _restart():
    return _load(_ROOT / "scripts" / "rolling_restart.py", "dipcatcher_rolling_restart")


def test_production_rate_must_be_set_above_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("RESEARCH_API_HOST", "127.0.0.1")
    monkeypatch.setenv("DIPCATCHER_BIND_HOST", "127.0.0.1")
    refused = "DIPCATCHER_RATE_LIMIT_RPS must be set above 0 in production"
    for profile in ("production", "prod", "Production"):
        monkeypatch.setenv("DIPCATCHER_PROFILE", profile)
        for raw in ("0", "0.0", "-1", "nope", "inf", "nan", "1000001"):
            if raw:
                monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", raw)
            else:
                monkeypatch.delenv("DIPCATCHER_RATE_LIMIT_RPS", raising=False)
            with pytest.raises(ServiceConfigError, match=refused) as raised:
                ServiceSettings.for_research()
            if raw not in {"0", "0.0"}:
                assert raw not in str(raised.value)
            assert "nope" not in str(raised.value)
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "50")
    assert ServiceSettings.for_research().service == "research"
    assert ServiceSettings.for_lab("127.0.0.1", 8000).service == "lab"
    monkeypatch.setenv("DIPCATCHER_PROFILE", "lab")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0")
    assert ServiceSettings.for_research().service == "research"
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    monkeypatch.delenv("DIPCATCHER_RATE_LIMIT_RPS", raising=False)
    assert ServiceSettings.for_research().service == "research"


def test_production_entrypoint_refuses_a_zero_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "0")
    called: list[str] = []

    def _exec(cmd: str, args: list[str]) -> None:
        called.append(cmd)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    try:
        assert entry.main(["entrypoint", "python", "-m", "quant_fund.api.research_api"]) == 1
    finally:
        data.chmod(0o755)
    assert called == []
    assert capsys.readouterr().err.strip() == (
        "DIPCATCHER_RATE_LIMIT_RPS must be set above 0 in production"
    )


def test_production_entrypoint_fails_closed_on_a_read_only_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "prod")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "10")
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "0")
    try:
        assert entry.main(["entrypoint", "python"]) == 1
    finally:
        data.chmod(0o755)
    message = capsys.readouterr().err.strip()
    assert message == (
        f"data directory {data} is not writable by uid {os.geteuid()}; "
        "the dipcatcher user is uid 10001"
    )
    assert "DIPCATCHER_RATE_LIMIT_RPS" not in message


def test_non_production_skips_the_write_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "0")
    seen: list[list[str]] = []

    def _exec(cmd: str, args: list[str]) -> None:
        seen.append(args)
        raise SystemExit(0)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    try:
        with pytest.raises(SystemExit) as raised:
            entry.main(["entrypoint", "python", "-m", "quant_fund.api.research_api"])
    finally:
        data.chmod(0o755)
    assert raised.value.code == 0
    assert seen == [["python", "-m", "quant_fund.api.research_api"]]


def test_prod_env_is_the_production_default_and_holds_no_secrets() -> None:
    env = (_ROOT / "deploy" / "prod.env").read_text(encoding="utf-8")
    assert "DIPCATCHER_PROFILE=production" in env
    assert "DIPCATCHER_RATE_LIMIT_RPS=50" in env
    assert "DIPCATCHER_REQUIRE_DATA_WRITE=1" in env
    assert "API_KEY=" not in env
    assert "DIPCATCHER_PROFILE" not in (_ROOT / ".env.example").read_text(encoding="utf-8")
    research = (_ROOT / "docker" / "research-api.Dockerfile").read_text(encoding="utf-8")
    lab = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "DIPCATCHER_REQUIRE_DATA_WRITE=0" in research
    assert "DIPCATCHER_REQUIRE_DATA_WRITE=1" in lab
    compose = yaml.safe_load((_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {
        "lab-a",
        "lab-b",
        "lab-canary",
        "research-a",
        "research-b",
        "proxy",
        "prometheus",
        "grafana",
        "node-exporter",
    }
    rendered = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    assert "GF_SECURITY_ADMIN_PASSWORD" not in rendered
    assert "9464:9464" not in rendered
    assert "0.0.0.0:" not in rendered
    assert "DIPCATCHER_METRICS_KEY_ENV: QUANT_API_KEY" in rendered
    assert "DIPCATCHER_METRICS_KEY_ENV: RESEARCH_API_KEY" in rendered
    rules = yaml.safe_load(
        (_ROOT / "deploy" / "observability" / "alerts.yml").read_text(encoding="utf-8")
    )
    expressions = [rule["expr"] for rule in rules["groups"][0]["rules"]]
    assert 'up{job=~"lab-api|research-api"} == 0' in expressions


def test_lab_cli_hook_authenticates_metrics_with_the_lab_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error
    import urllib.request

    from quant_fund.cli.main import _maybe_install_observation_hooks
    from quant_fund.observe.install import uninstall_passive_hooks
    from quant_fund.observe.serve import stop_metrics_server

    uninstall_passive_hooks()
    stop_metrics_server()
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("DIPCATCHER_METRICS_HOST", "127.0.0.1")
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "0")
    monkeypatch.setenv("QUANT_API_KEY", "metrics-test-key-0123456789")
    monkeypatch.delenv("DIPCATCHER_METRICS_KEY_ENV", raising=False)
    monkeypatch.delenv("DIPCATCHER_METRICS_LOOPBACK_SCRAPE", raising=False)
    try:
        _maybe_install_observation_hooks()
        from quant_fund.observe import serve as metrics_serve

        assert metrics_serve._SERVER is not None
        port = int(metrics_serve._SERVER.server_address[1])
        url = f"http://127.0.0.1:{port}/metrics"

        def _status(key: str | None) -> int:
            request = urllib.request.Request(url)
            if key is not None:
                request.add_header("Authorization", f"Bearer {key}")
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    response.read()
                    return int(response.status)
            except urllib.error.HTTPError as exc:
                return int(exc.code)

        assert _status(None) == 401
        assert _status("metrics-test-key-0123456789") == 200
        assert os.environ["DIPCATCHER_METRICS_KEY_ENV"] == "QUANT_API_KEY"
    finally:
        uninstall_passive_hooks()
        stop_metrics_server()


def test_rolling_restart_accepts_drain_and_rejects_5xx() -> None:
    module = _restart()
    assert module.drain_observed("503")
    assert module.drain_observed("transport")
    assert not module.drain_observed("200")
    assert not module.drain_observed("401")
    assert module.load_has_5xx({"http_5xx": 2})
    assert module.load_has_5xx({})
    assert not module.load_has_5xx({"http_5xx": 0})

"""Production file secrets, request ids, SLO burn rate, and rollback."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from quant_fund.api.rate_limit import reset_bucket
from quant_fund.api.secret_files import SecretFileError, apply_secret_files, clear_loaded_secrets
from quant_fund.api.service_config import ServiceConfigError, ServiceSettings
from quant_fund.observe.metrics import render_prometheus, reset_metrics

_ROOT = Path(__file__).resolve().parents[3]
_KEY = "planted-test-key-value-0123456789"
_OTHER = "metrics-test-key-0123456789"
_WEBHOOK = "webhook-test-key-0123456789"
_SENT = "ops10-lab-0123456789abcdef"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DIPCATCHER_PROFILE",
        "DIPCATCHER_ALERT_WEBHOOK_URL",
        "DIPCATCHER_ALERT_WEBHOOK_SECRET",
        "DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE",
        "QUANT_API_KEY",
        "QUANT_API_KEY_FILE",
        "RESEARCH_API_KEY",
        "RESEARCH_API_KEY_FILE",
        "DIPCATCHER_RATE_LIMIT_RPS",
        "DIPCATCHER_OBSERVE",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_loaded_secrets()
    reset_metrics()
    reset_bucket()
    yield
    clear_loaded_secrets()
    reset_metrics()
    reset_bucket()


def _write(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o600)


def test_production_plain_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    monkeypatch.setenv("DIPCATCHER_BIND_HOST", "127.0.0.1")
    with pytest.raises(ServiceConfigError, match="QUANT_API_KEY_FILE") as raised:
        ServiceSettings.for_lab("127.0.0.1", 8000)
    assert _KEY not in str(raised.value)


def test_production_file_key_boots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "quant_api_key"
    _write(path, _KEY)
    monkeypatch.setenv("DIPCATCHER_PROFILE", " Prod ")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(path))
    monkeypatch.setenv("DIPCATCHER_BIND_HOST", "127.0.0.1")
    settings = ServiceSettings.for_lab("127.0.0.1", 8000)
    assert settings.api_key_configured is True
    assert os.environ["QUANT_API_KEY"] == _KEY
    again = ServiceSettings.for_lab("127.0.0.1", 8000)
    assert again.service == "lab"


def test_file_and_plain_key_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "quant_api_key"
    _write(path, _KEY)
    monkeypatch.setenv("QUANT_API_KEY", _OTHER)
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(path))
    with pytest.raises(ServiceConfigError, match="must not be set") as raised:
        ServiceSettings.for_lab("127.0.0.1", 8000)
    assert _KEY not in str(raised.value)
    assert _OTHER not in str(raised.value)


def test_non_production_plain_key_still_boots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    assert ServiceSettings.for_lab("127.0.0.1", 8000).api_key_configured is True


def test_webhook_file_is_required_when_the_url_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_ALERT_WEBHOOK_URL", "http://alerts.example/hook")
    problems = apply_secret_files()
    assert any("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE is required" in item for item in problems)
    path = tmp_path / "hook"
    _write(path, _WEBHOOK)
    monkeypatch.setenv("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", str(path))
    clear_loaded_secrets()
    assert apply_secret_files() == []
    assert os.environ["DIPCATCHER_ALERT_WEBHOOK_SECRET"] == _WEBHOOK
    monkeypatch.setenv(
        "DIPCATCHER_ALERT_WEBHOOK_URL", "https://user:webhook-test-key@alerts.example/hook"
    )
    clear_loaded_secrets()
    monkeypatch.delenv("DIPCATCHER_ALERT_WEBHOOK_SECRET", raising=False)
    refused = apply_secret_files()
    assert refused == ["DIPCATCHER_ALERT_WEBHOOK_URL is invalid"]
    assert "webhook-test-key" not in " ".join(refused)


def test_symlink_secret_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "real"
    _write(target, _KEY)
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(link))
    with pytest.raises(ServiceConfigError, match="not a readable file"):
        ServiceSettings.for_lab("127.0.0.1", 8000)


def test_entrypoint_refuses_a_plain_production_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _load(_ROOT / "docker" / "entrypoint.py", "dipcatcher_entrypoint_ops10")
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "50")
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    monkeypatch.setattr(entry.os, "execvp", lambda *_args: (_ for _ in ()).throw(AssertionError))
    assert entry.main(["entrypoint", "dipcatcher", "api"]) == 1
    message = capsys.readouterr().err
    assert "QUANT_API_KEY_FILE" in message
    assert _KEY not in message


def test_entrypoint_leaves_the_file_key_for_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _load(_ROOT / "docker" / "entrypoint.py", "dipcatcher_entrypoint_ops10b")
    data = tmp_path / "data"
    data.mkdir()
    secret = tmp_path / "quant_api_key"
    _write(secret, _KEY)
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "50")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(secret))
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    seen: list[str | None] = []

    def _exec(_cmd: str, _args: list[str]) -> None:
        seen.append(os.environ.get("QUANT_API_KEY"))
        raise SystemExit(0)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    with pytest.raises(SystemExit):
        entry.main(["entrypoint", "dipcatcher", "api"])
    assert seen == [None]
    assert _KEY not in capsys.readouterr().err


def test_request_log_propagates_the_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from quant_fund.api.request_log import RequestLogMiddleware

    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")

    async def app(scope: dict, receive, send) -> None:  # type: ignore[no-untyped-def]
        del scope, receive
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestLogMiddleware(app, service="lab")

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: dict[str, list[dict[str, object]]] = {}

    async def send(message: dict[str, object]) -> None:
        sent.setdefault("messages", []).append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [(b"x-request-id", _SENT.encode("ascii"))],
    }
    asyncio.run(middleware(scope, receive, send))
    headers = dict(sent["messages"][0]["headers"])
    assert headers[b"x-request-id"] == _SENT.encode("ascii")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "request"
    assert payload["request_id"] == _SENT
    assert payload["path"] == "/health"
    assert _KEY not in line
    text = render_prometheus()
    assert 'dipcatcher_http_requests_total{code="204",service="lab"} 1' in text


def test_request_log_replaces_an_unsafe_id(capsys: pytest.CaptureFixture[str]) -> None:
    from quant_fund.api.request_log import RequestLogMiddleware, accept_request_id

    assert accept_request_id("bad id") != "bad id"
    assert " " not in accept_request_id("bad id")

    async def app(scope: dict, receive, send) -> None:  # type: ignore[no-untyped-def]
        del scope, receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestLogMiddleware(app, service="research")

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    captured: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        captured.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [(b"x-request-id", b"bad id")],
    }
    asyncio.run(middleware(scope, receive, send))
    echoed = dict(captured[0]["headers"])[b"x-request-id"]
    assert b" " not in echoed
    assert b"bad id" not in capsys.readouterr().err.encode("utf-8")


def test_slo_rules_and_promtool() -> None:
    rules = (_ROOT / "deploy" / "observability" / "slo_rules.yml").read_text(encoding="utf-8")
    assert "dipcatcher:http_requests:availability_ratio_30d" in rules
    assert "dipcatcher:http_latency:p99_5m" in rules
    assert "dipcatcher:slo:error_budget_remaining_30d" in rules
    assert "DipcatcherErrorBudgetBurn" in rules
    prometheus = (_ROOT / "deploy" / "observability" / "prometheus.yml").read_text(encoding="utf-8")
    assert "slo_rules.yml" in prometheus
    if shutil.which("docker") is None and shutil.which("sudo") is None:
        pytest.skip("promtool is not available")
    proc = subprocess.run(
        [
            "sudo",
            "docker",
            "run",
            "--rm",
            "-v",
            f"{_ROOT / 'deploy' / 'observability'}:/rules:ro",
            "--entrypoint",
            "/bin/promtool",
            "prom/prometheus:v2.54.1",
            "test",
            "rules",
            "/rules/slo_rules_test.yml",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    docker_down = "Cannot connect" in proc.stderr or "not found" in proc.stderr
    if proc.returncode != 0 and "SUCCESS" not in proc.stdout and docker_down:
        pytest.skip("promtool is not available")
    assert proc.returncode == 0
    assert "SUCCESS" in proc.stdout


def test_compose_uses_file_secrets_and_rollback_is_in_deploy_mk() -> None:
    text = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert set(loaded["secrets"]) == {
        "quant_api_key",
        "research_api_key",
        "alert_webhook_secret",
    }
    lab = loaded["services"]["lab-a"]["environment"]
    assert lab["QUANT_API_KEY_FILE"] == "/run/secrets/quant_api_key"
    assert "QUANT_API_KEY" not in lab
    research = loaded["services"]["research-a"]["environment"]
    assert research["RESEARCH_API_KEY_FILE"] == "/run/secrets/research_api_key"
    assert "RESEARCH_API_KEY" not in research
    assert loaded["services"]["proxy"]["entrypoint"][0] == "/bin/sh"
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "prod-rollback:" in prod
    assert "dipcatcher:previous" in prod
    assert "prod-rollback:" not in root
    assert "API_KEY=" not in (_ROOT / "deploy" / "prod.env.example").read_text(encoding="utf-8")
    runbook = (_ROOT / "docs" / "OPS_RUNBOOK_PROD.md").read_text(encoding="utf-8")
    assert "## Secrets" in runbook
    assert "## Request ids" in runbook
    assert "## Service level objectives" in runbook
    assert "make -f deploy/prod.mk prod-rollback" in runbook


def test_request_id_and_rollback_gate() -> None:
    smoke = _load(_ROOT / "scripts" / "prod_smoke.py", "dipcatcher_prod_smoke_ops10")
    gate = _load(_ROOT / "scripts" / "prod_rollback.py", "dipcatcher_prod_rollback_ops10")
    sent = _SENT
    assert smoke.request_id_visible(sent, sent, f"header {sent}", f'"request_id": "{sent}"')
    assert not smoke.request_id_visible(sent, "other", sent, sent)
    assert not smoke.request_id_visible(sent, sent, "proxy", "app")
    assert gate.listener_ok(
        200, {"Strict-Transport-Security": "max-age=31536000; includeSubDomains"}
    )
    assert not gate.listener_ok(200, {})
    assert gate.gate_ready(["healthy", "healthy", "healthy", "healthy"], True, True)
    assert not gate.gate_ready(["healthy", "starting", "healthy", "healthy"], True, True)
    assert not gate.gate_ready(["healthy"], True, True)


def test_secret_files_are_readable_by_the_container_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load(_ROOT / "scripts" / "prod_secrets.py", "dipcatcher_prod_secrets_ops10")
    quant = tmp_path / "quant_api_key"
    research = tmp_path / "research_api_key"
    webhook = tmp_path / "alert_webhook_secret"
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    monkeypatch.setenv("RESEARCH_API_KEY", _OTHER)
    monkeypatch.delenv("DIPCATCHER_ALERT_WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("DIPCATCHER_QUANT_API_KEY_FILE", str(quant))
    monkeypatch.setenv("DIPCATCHER_RESEARCH_API_KEY_FILE", str(research))
    monkeypatch.setenv("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", str(webhook))
    assert writer.main(["prod_secrets.py"]) == 0
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert quant.read_text(encoding="utf-8").strip() == _KEY
    assert research.read_text(encoding="utf-8").strip() == _OTHER
    token = webhook.read_text(encoding="utf-8").strip()
    assert token and " " not in token and _KEY not in token
    for path in (quant, research, webhook):
        assert path.stat().st_mode & 0o777 == 0o644
    webhook.chmod(0o600)
    assert writer.main(["prod_secrets.py"]) == 0
    assert webhook.stat().st_mode & 0o777 == 0o644
    assert webhook.read_text(encoding="utf-8").strip() == token


def test_research_settings_load_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.api.research_api import ResearchApiSettings

    path = tmp_path / "research_api_key"
    _write(path, _OTHER)
    monkeypatch.setenv("DIPCATCHER_PROFILE", "prod")
    monkeypatch.setenv("RESEARCH_API_KEY_FILE", str(path))
    settings = ResearchApiSettings.from_env()
    assert settings.api_key == _OTHER
    clear_loaded_secrets()
    monkeypatch.setenv("RESEARCH_API_KEY", _KEY)
    with pytest.raises(SecretFileError, match="must not be set"):
        ResearchApiSettings.from_env()

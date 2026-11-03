"""Alert routing, secret overlap, and soak slope checks."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.secret_files import apply_secret_files, clear_loaded_secrets
from quant_fund.api.webhook_hmac import sign_alert_webhook, verify_alert_webhook
from quant_fund.live.brokers import AlpacaBroker, alpaca_credentials, broker_overlap_pairs
from quant_fund.live.errors import LiveBrokerRefused, LiveGateError
from quant_fund.utils.secret_tokens import (
    accepted_tokens,
    auth_decision,
    parse_overlap,
    read_secret_tokens,
)

_ROOT = Path(__file__).resolve().parents[3]
_OLD = "test-key-old"
_NEW = "test-key-new"
_OTHER = "test-key-other"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_parse_overlap_accepts_one_or_two_lines() -> None:
    assert parse_overlap(f"{_OLD}\n") == (_OLD,)
    assert parse_overlap(f"{_OLD}\n{_NEW}\n") == (_OLD, _NEW)
    assert parse_overlap(f"{_OLD}\n\n{_NEW}") == (_OLD, _NEW)
    assert parse_overlap(f"{_OLD} {_NEW}") is None
    assert parse_overlap(f"{_OLD}\n{_NEW}\n{_OTHER}") is None


def test_file_auth_accepts_both_lines_and_rejects_a_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "quant_api_key"
    path.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(path))
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    assert accepted_tokens("QUANT_API_KEY") == (_OLD, _NEW)
    assert auth_decision("QUANT_API_KEY", _OLD) == "ok"
    assert auth_decision("QUANT_API_KEY", _NEW) == "ok"
    assert auth_decision("QUANT_API_KEY", _OTHER) == "reject"
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(tmp_path))
    assert auth_decision("QUANT_API_KEY", _OLD) == "reject"
    from quant_fund.api.app import app

    client = TestClient(app)
    assert client.get("/ready", headers={"X-API-Key": _OLD}).status_code == 401
    path.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(path))
    assert client.get("/ready", headers={"X-API-Key": _NEW}).status_code != 401
    assert client.get("/ready", headers={"X-API-Key": _OLD}).status_code != 401


def test_secret_reads_do_not_grow_file_descriptors(tmp_path: Path) -> None:
    path = tmp_path / "secret"
    path.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    before = _fds()
    for _ in range(200):
        assert read_secret_tokens(str(path)) == (_OLD, _NEW)
    assert _fds() <= before + 1


def test_research_file_accepts_both_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.api.research_api import ResearchApiSettings, create_app

    path = tmp_path / "research_api_key"
    path.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_API_KEY_FILE", str(path))
    monkeypatch.delenv("RESEARCH_API_KEY", raising=False)
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
    )
    for directory in (
        settings.data_root,
        settings.receipts_dir,
        settings.verifier_dir,
        settings.artifacts_dir,
    ):
        directory.mkdir()
    client = TestClient(create_app(settings), client=("203.0.113.8", 9))
    assert client.get("/receipts", headers={"X-API-Key": _OLD}).status_code != 401
    assert client.get("/receipts", headers={"X-API-Key": _NEW}).status_code != 401
    assert client.get("/receipts").status_code == 401


def test_boot_loads_the_first_of_two_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "quant_api_key"
    path.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(path))
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    clear_loaded_secrets()
    assert apply_secret_files() == []
    assert os.environ["QUANT_API_KEY"] == _OLD
    clear_loaded_secrets()


def test_entrypoint_accepts_two_tokens_and_refuses_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _load(_ROOT / "docker" / "entrypoint.py", "dipcatcher_entrypoint_ops13")
    data = tmp_path / "data"
    data.mkdir()
    secret = tmp_path / "quant_api_key"
    secret.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "50")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(secret))
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.setattr(entry.os, "execvp", lambda *_args: (_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit) as raised:
        entry.main(["entrypoint", "dipcatcher", "api"])
    assert raised.value.code == 0
    secret.write_text(f"{_OLD}\n{_NEW}\n{_OTHER}\n", encoding="utf-8")
    assert entry.main(["entrypoint", "dipcatcher", "api"]) == 1


def test_proxy_entrypoint_loads_line_one_and_rejects_a_bad_line(tmp_path: Path) -> None:
    script = (_ROOT / "deploy" / "proxy" / "secret-entrypoint.sh").read_text(encoding="utf-8")
    captured = tmp_path / "out"
    fake = tmp_path / "caddy"
    fake.write_text(f"#!/bin/sh\nprintf '%s' \"$QUANT_API_KEY\" > {captured}\n", encoding="utf-8")
    fake.chmod(0o755)
    patched = script.replace("exec /usr/bin/caddy", f"exec {fake}")
    runner = tmp_path / "entry.sh"
    runner.write_text(patched, encoding="utf-8")
    secret = tmp_path / "quant_api_key"
    secret.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    env = os.environ.copy()
    env["QUANT_API_KEY_FILE"] = str(secret)
    env.pop("QUANT_API_KEY", None)
    env.pop("RESEARCH_API_KEY", None)
    env.pop("RESEARCH_API_KEY_FILE", None)
    env.pop("DIPCATCHER_PROFILE", None)
    done = subprocess.run(
        ["sh", str(runner), "caddy", "run"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert captured.read_text(encoding="utf-8") == _OLD
    assert _OLD not in done.stderr
    secret.write_text(f"{_OLD}\nbad token\n", encoding="utf-8")
    rejected = subprocess.run(
        ["sh", str(runner), "caddy", "run"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert "one or two tokens" in rejected.stderr


def test_webhook_overlap_verifies_both_secrets() -> None:
    now = 1_700_000_000.0
    body = b"{}"
    stamp = str(int(now))
    old_sig = sign_alert_webhook(_OLD, stamp, body)
    new_sig = sign_alert_webhook(_NEW, stamp, body)
    assert verify_alert_webhook((_OLD, _NEW), stamp, old_sig, body, now=now)
    assert verify_alert_webhook((_OLD, _NEW), stamp, new_sig, body, now=now)
    assert not verify_alert_webhook((_NEW,), stamp, old_sig, body, now=now)
    assert not verify_alert_webhook((_OLD, _NEW), stamp, new_sig, body, now=now + 301)


def test_broker_retries_401_once_and_refuses_live(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key-a")
    with pytest.raises(LiveGateError, match="incomplete"):
        alpaca_credentials()
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-key-asec")
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-b")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-key-bsec")
    assert broker_overlap_pairs()[0][0] == "test-key-a"
    assert broker_overlap_pairs()[1][0] == "test-key-b"
    calls: list[str] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, body, timeout
        calls.append(headers["APCA-API-KEY-ID"])
        if len(calls) == 1:
            return 401, b""
        if url.endswith("/account"):
            return 200, json.dumps({"cash": "1", "equity": "2"}).encode()
        return 200, b"[]"

    broker = AlpacaBroker(
        "alpaca-paper",
        ("test-key-a", "test-key-asec"),
        fallback=("test-key-b", "test-key-bsec"),
        transport=transport,
    )
    assert broker.account().cash == 1
    assert calls[:2] == ["test-key-a", "test-key-b"]
    calls.clear()

    def ok_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, body, timeout
        calls.append(headers["APCA-API-KEY-ID"])
        if url.endswith("/account"):
            return 200, json.dumps({"cash": "3", "equity": "4"}).encode()
        return 200, b"[]"

    quiet = AlpacaBroker(
        "alpaca-paper",
        ("test-key-a", "test-key-asec"),
        fallback=("test-key-b", "test-key-bsec"),
        transport=ok_transport,
    )
    assert quiet.account().cash == 3
    assert calls == ["test-key-a", "test-key-a"]
    live_calls: list[str] = []

    def live_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        live_calls.append("hit")
        return 200, b"{}"

    live = AlpacaBroker(
        "alpaca-live",
        ("test-key-a", "test-key-asec"),
        fallback=("test-key-b", "test-key-bsec"),
        transport=live_transport,
    )
    with pytest.raises(LiveBrokerRefused, match="alpaca-live"):
        live.account()
    assert live_calls == []


def test_rotate_drill_and_fail_closed_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(_ROOT / "scripts" / "prod_rotate.py", "dipcatcher_prod_rotate_ops13")
    monkeypatch.setenv("DIPCATCHER_QUANT_API_KEY_FILE", str(tmp_path / "quant_api_key"))
    monkeypatch.setenv("DIPCATCHER_RESEARCH_API_KEY_FILE", str(tmp_path / "research_api_key"))
    monkeypatch.setenv("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", str(tmp_path / "hook"))
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_API_KEY", raising=False)
    monkeypatch.delenv("DIPCATCHER_ALERT_WEBHOOK_SECRET", raising=False)
    refused = module._apply_phase("overlap")
    assert refused["ok"] is False
    assert refused["error"] == "QUANT_API_KEY"
    (tmp_path / "quant_api_key").write_text(_OLD + "\n", encoding="utf-8")
    (tmp_path / "research_api_key").write_text(_OTHER + "\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY", _NEW)
    monkeypatch.setenv("RESEARCH_API_KEY", "test-key-next")
    report = module._apply_phase("overlap")
    assert report["ok"] is True
    assert (tmp_path / "quant_api_key").read_text(encoding="utf-8") == f"{_OLD}\n{_NEW}\n"
    promoted = module._apply_phase("promote")
    assert promoted["ok"] is True
    assert (tmp_path / "quant_api_key").read_text(encoding="utf-8") == f"{_NEW}\n{_OLD}\n"
    drill = module.run_drill()
    assert drill["ok"] is True
    assert drill["gap"] is False
    assert drill["api_old_during_overlap"] is True
    assert drill["api_new_during_overlap"] is True
    assert drill["api_old_after_retire"] is True
    assert drill["webhook_old_during_overlap"] is True
    assert drill["webhook_old_after_retire"] is False
    assert drill["broker_pairs_during_overlap"] == 2
    assert drill["broker_pairs_after_retire"] == 1


def test_prod_secrets_keeps_an_overlap_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load(_ROOT / "scripts" / "prod_secrets.py", "dipcatcher_prod_secrets_ops13")
    quant = tmp_path / "quant_api_key"
    research = tmp_path / "research_api_key"
    webhook = tmp_path / "alert_webhook_secret"
    quant.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    research.write_text("test-key-next\n", encoding="utf-8")
    webhook.write_text(f"{_OLD}\n{_NEW}\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY", _NEW)
    monkeypatch.setenv("RESEARCH_API_KEY", "test-key-next")
    monkeypatch.delenv("DIPCATCHER_ALERT_WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("DIPCATCHER_QUANT_API_KEY_FILE", str(quant))
    monkeypatch.setenv("DIPCATCHER_RESEARCH_API_KEY_FILE", str(research))
    monkeypatch.setenv("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", str(webhook))
    assert writer.main(["prod_secrets.py"]) == 0
    assert quant.read_text(encoding="utf-8") == f"{_OLD}\n{_NEW}\n"
    assert webhook.read_text(encoding="utf-8") == f"{_OLD}\n{_NEW}\n"
    monkeypatch.setenv("QUANT_API_KEY", "test-key-third")
    assert writer.main(["prod_secrets.py"]) == 0
    assert quant.read_text(encoding="utf-8") == "test-key-third\n"


def test_alert_inventory_routes_dedupes_and_resolves() -> None:
    module = _load(_ROOT / "scripts" / "prod_alert_drill.py", "dipcatcher_prod_alert_drill_ops13")
    names = module.alert_names(
        (_ROOT / "deploy" / "observability" / "alerts.yml").read_text(encoding="utf-8"),
        (_ROOT / "deploy" / "observability" / "slo_rules.yml").read_text(encoding="utf-8"),
    )
    for required in module._REQUIRED:
        assert required in names
    assert "DipcatcherTargetDown" in names
    assert "DipcatcherErrorBudgetBurn" in names
    fault = module.fault_metrics(4)
    assert "dipcatcher_kill_switch_blocking 1" in fault
    assert "dipcatcher_reconcile_drift 1" in fault
    assert "dipcatcher_backup_last_success_timestamp 0" in fault
    assert "dipcatcher_canary_abort 1" in fault
    assert 'node_filesystem_avail_bytes{device="/dev/drill"' in fault
    healthy = module.healthy_metrics(1_700_000_000, 200_000)
    assert "dipcatcher_ready 1" in healthy
    assert "dipcatcher_errors_total" not in healthy
    assert 'dipcatcher_http_requests_total{code="200",service="lab"} 200000' in healthy
    events = [(name, f"fp-{name}", "firing") for name in names]
    open_verdict = module.notification_verdict(names, events)
    assert open_verdict["resolved"] is False
    closed = events + [(name, f"fp-{name}", "resolved") for name in names]
    verdict = module.notification_verdict(names, closed)
    assert verdict["deduped"] is True
    assert verdict["resolved"] is True
    assert verdict["missing"] == []
    duplicated = closed + [(names[0], f"fp-{names[0]}", "firing")]
    assert module.notification_verdict(names, duplicated)["deduped"] is False
    config = module.prometheus_config(1, 2, 3, 4)
    assert config.count("job_name:") == 2
    assert "127.0.0.1:1" in config and "127.0.0.1:3" in config


def test_soak_slope_flags_a_climb_and_ignores_warmup() -> None:
    module = _load(_ROOT / "scripts" / "prod_soak.py", "dipcatcher_prod_soak_ops13")
    assert module.slope([(0.0, 0.0), (10.0, 10.0)]) == pytest.approx(1.0)
    assert module.slope([(0.0, 1.0)]) == 0.0
    flat = [
        {"t": 0.0, "fds": 10.0, "rss_kb": 1000.0},
        {"t": 130.0, "fds": 10.0, "rss_kb": 1000.0},
        {"t": 160.0, "fds": 10.0, "rss_kb": 1000.0},
        {"t": 190.0, "fds": 12.0, "rss_kb": 1000.0},
    ]
    assert module.leak_decision(flat)["leak"] is False
    climbing = [
        {"t": 0.0, "fds": 10.0, "rss_kb": 1000.0},
        {"t": 130.0, "fds": 10.0, "rss_kb": 1000.0},
        {"t": 160.0, "fds": 40.0, "rss_kb": 1000.0},
        {"t": 190.0, "fds": 70.0, "rss_kb": 1000.0},
    ]
    assert module.leak_decision(climbing)["leak"] is True


def test_textfile_signals_and_publishers() -> None:
    signals = _load(_ROOT / "scripts" / "ops_signals.py", "dipcatcher_ops_signals_ops13")
    directory = Path("/tmp/dipcatcher-ops13-signals")
    if directory.exists():
        for child in directory.iterdir():
            child.unlink()
    try:
        assert signals.write_prom(directory, {"dipcatcher_canary_abort": 1}) is None
        assert signals.read_gauges(directory / "dipcatcher.prom")["dipcatcher_canary_abort"] == 1
        assert (
            signals.write_prom(directory, {"dipcatcher_backup_last_success_timestamp": 10}) is None
        )
        gauges = signals.read_gauges(directory / "dipcatcher.prom")
        assert gauges["dipcatcher_canary_abort"] == 1
        assert gauges["dipcatcher_backup_last_success_timestamp"] == 10
        assert (
            signals.write_prom(Path("/tmp/not-dipcatcher-ops13"), {"dipcatcher_canary_abort": 1})
            == "path"
        )
        backup = _load(_ROOT / "scripts" / "prod_backup_ship.py", "dipcatcher_backup_ship_ops13")
        backup._publish_backup({"DIPCATCHER_TEXTFILE_DIR": str(directory)})
        stamped = signals.read_gauges(directory / "dipcatcher.prom")
        assert stamped["dipcatcher_backup_last_success_timestamp"] > 10
        canary = _load(_ROOT / "scripts" / "prod_canary.py", "dipcatcher_prod_canary_ops13")
        canary._publish_canary({"DIPCATCHER_TEXTFILE_DIR": str(directory)}, 1)
        assert signals.read_gauges(directory / "dipcatcher.prom")["dipcatcher_canary_abort"] == 1
        canary._publish_canary({"DIPCATCHER_TEXTFILE_DIR": ""}, 0)
        assert signals.read_gauges(directory / "dipcatcher.prom")["dipcatcher_canary_abort"] == 1
    finally:
        if directory.exists():
            for child in directory.iterdir():
                child.unlink()
            directory.rmdir()


def test_prod_targets_textfile_and_runbook() -> None:
    makefile = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    compose = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    runbook = (_ROOT / "docs" / "OPS_RUNBOOK_PROD.md").read_text(encoding="utf-8")
    checklist = (_ROOT / "docs" / "PRODUCTION.md").read_text(encoding="utf-8")
    example = (_ROOT / "deploy" / "prod.env.example").read_text(encoding="utf-8")
    assert "prod-alert-drill:" in makefile
    assert "prod-rotate-drill:" in makefile
    assert "prod-soak:" in makefile
    assert "--collector.textfile.directory=/textfile" in compose
    assert "0.0.0.0:" not in compose
    assert "## Alert drill" in runbook
    assert "## Secret rotation" in runbook
    assert "## Capacity soak" in runbook
    assert "Re-sign" in checklist
    assert "fx-1 weights" in checklist
    assert "virtualenv" in checklist
    assert "TEXTFILE_DIR" in example
    assert "line 2" in example

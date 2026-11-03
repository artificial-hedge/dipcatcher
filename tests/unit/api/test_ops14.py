"""Rollback flag, gauge refresh, redaction, runbooks, and order-path gates."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from quant_fund.api.json_logging import JsonLineFormatter, emit_service_event, redact_text

_ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _names(path: Path) -> list[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for group in document["groups"]:
        for rule in group["rules"]:
            if "alert" in rule:
                found.append(rule["alert"])
    return found


def test_every_alert_links_a_runbook() -> None:
    paths = [
        _ROOT / "deploy" / "observability" / "alerts.yml",
        _ROOT / "deploy" / "observability" / "slo_rules.yml",
    ]
    names: list[str] = []
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for group in document["groups"]:
            for rule in group["rules"]:
                if "alert" not in rule:
                    continue
                name = rule["alert"]
                names.append(name)
                runbook = rule["annotations"]["runbook"]
                assert runbook == f"docs/incidents/{name}.md"
                text = (_ROOT / runbook).read_text(encoding="utf-8")
                for heading in (
                    "## Trigger",
                    "## Impact",
                    "## First 5 minutes",
                    "## Rollback and kill switch",
                ):
                    assert heading in text
    assert len(names) == 17
    assert len(set(names)) == 17
    assert "DipcatcherTargetDown" in names
    index = (_ROOT / "docs" / "incidents" / "README.md").read_text(encoding="utf-8")
    for name in names:
        assert f"({name}.md)" in index


def test_refresh_republishes_sidecars() -> None:
    signals = _load(_ROOT / "scripts" / "ops_signals.py", "dipcatcher_ops_signals_ops14")
    directory = Path("/tmp/dipcatcher-ops14-unit-signals")
    if directory.exists():
        shutil.rmtree(directory)
    signals.publish(
        str(directory),
        {
            "dipcatcher_canary_abort": 1,
            "dipcatcher_backup_last_success_timestamp": 10,
            "dipcatcher_reconcile_drift": 0,
        },
    )
    assert signals.read_state(directory, "canary.state") == 1
    assert signals.read_state(directory, "backup.stamp") == 10
    gauges = signals.refresh_gauges(directory, ["docs"])
    assert gauges["dipcatcher_reconcile_drift"] == 1
    assert gauges["dipcatcher_canary_abort"] == 1
    assert gauges["dipcatcher_backup_last_success_timestamp"] == 10
    signals.publish(str(directory), {"dipcatcher_canary_abort": 0})
    assert signals.refresh_gauges(directory, [])["dipcatcher_canary_abort"] == 0
    assert signals.refresh_gauges(directory, [])["dipcatcher_backup_last_success_timestamp"] == 10


def test_failed_canary_rolls_the_previous_images_back() -> None:
    module = _load(_ROOT / "scripts" / "prod_canary.py", "dipcatcher_prod_canary_ops14")
    env = {
        "QUANT_API_KEY": "test-key-lab",
        "RESEARCH_API_KEY": "test-key-research",
        "SUDO": "",
        "DIPCATCHER_AUTO_ROLLBACK": "1",
    }
    calls: list[list[str]] = []

    class _Done:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""

    def _run(argv: list[str]) -> _Done:
        calls.append(list(argv))
        return _Done()

    report = module.execute(env, _run, sleep=lambda _seconds: None, attempts=1)
    assert report["action"] == "abort"
    assert report["rollback_ok"] is True
    assert report["ok"] is False
    flat = [" ".join(item) for item in calls]
    assert any("tag dipcatcher:previous dipcatcher:local" in item for item in flat)
    assert any("dipcatcher-research-api:previous" in item and "tag" in item for item in flat)
    assert any("--force-recreate" in item and "lab-canary" not in item for item in flat)
    plain = dict(env)
    plain["DIPCATCHER_AUTO_ROLLBACK"] = "0"
    quiet: list[list[str]] = []
    stopped = module.execute(plain, lambda argv: quiet.append(list(argv)) or _Done(), attempts=1)
    assert stopped["rollback_ok"] is False
    assert not any("tag" in item for item in quiet)


def test_redaction_scanner_fails_on_a_planted_leak(tmp_path: Path) -> None:
    audit = _load(_ROOT / "scripts" / "redaction_audit.py", "dipcatcher_redaction_audit_ops14")
    secret = "test-key-planted"
    assert audit.findings(f"saw {secret}", (secret,)) == ["needle"]
    assert "email" in audit.findings("user@example.com", ())
    assert "ssn" in audit.findings("id 123-45-6789", ())
    assert "bearer" in audit.findings("Authorization: Bearer abcdefghijklmnop", ())
    assert "account_label" in audit.findings('account_id="ABC123"', ())
    assert audit.findings("healthy lab", (secret,)) == []
    leaked = tmp_path / "proxy.log"
    leaked.write_text(f"token {secret}\n", encoding="utf-8")
    assert audit.scan_files([leaked], (secret,))[0]["kinds"] == ["needle"]
    clean = tmp_path / "clean.log"
    clean.write_text("request_id=abc\n", encoding="utf-8")
    assert audit.scan_files([clean], (secret,)) == []


def test_formatter_redacts_file_tokens_and_broker_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = tmp_path / "quant_api_key"
    secret.write_text("test-key-current\ntest-key-previous\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(secret))
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-key-broker")
    text = "current test-key-current previous test-key-previous broker test-key-broker"
    redacted = redact_text(text)
    assert "test-key-current" not in redacted
    assert "test-key-previous" not in redacted
    assert "test-key-broker" not in redacted
    assert redacted.count("[redacted]") == 3
    emit_service_event("probe", detail=text)
    logged = capsys.readouterr().err
    assert "test-key-current" not in logged
    record = type("R", (), {})()
    record.levelname = "INFO"
    record.name = "dipcatcher"
    record.getMessage = lambda: text
    record.exc_info = None
    rendered = JsonLineFormatter().format(record)
    assert "test-key-previous" not in rendered


def test_order_load_reads_the_production_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    monkeypatch.delenv("DIPCATCHER_WRITE_OWNER", raising=False)
    monkeypatch.delenv("DIPCATCHER_KILL_SWITCH_PATH", raising=False)
    module = _load(_ROOT / "scripts" / "prod_order_load.py", "dipcatcher_prod_order_load_ops14")
    assert module.expected_rps("DIPCATCHER_RATE_LIMIT_RPS=50\n") == 50
    assert module.expected_rps("other=1\n") == 0
    paper = module._paper_burst(tmp_path, 40)
    assert paper["notional_reason"] == "max_order_notional"
    assert paper["accepted_over_notional"] == 0
    assert paper["kill_accepted"] == 0
    assert float(paper["rps"]) > 0
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    for name in (
        "prod-rollback-drill",
        "prod-signal-drill",
        "prod-order-load",
        "prod-redaction-audit",
    ):
        assert f"{name}:" in prod
        assert f"{name}:" not in (_ROOT / "Makefile").read_text(encoding="utf-8")
    unit = (_ROOT / "deploy" / "systemd" / "dipcatcher-ops-signals.service").read_text(
        encoding="utf-8"
    )
    assert "ops_signals.py refresh" in unit
    assert "ops_signals.py refresh" in (
        _ROOT / "deploy" / "cron" / "dipcatcher-ops-signals"
    ).read_text(encoding="utf-8")


def test_signal_case_uses_published_faults() -> None:
    module = _load(_ROOT / "scripts" / "prod_signal_drill.py", "dipcatcher_prod_signal_drill_ops14")
    directory = Path("/tmp/dipcatcher-ops14-unit-faults")
    if directory.exists():
        shutil.rmtree(directory)
    gauges = module.publish_faults(directory)
    assert gauges["dipcatcher_reconcile_drift"] == 1
    assert gauges["dipcatcher_backup_last_success_timestamp"] == 0
    assert gauges["dipcatcher_canary_abort"] == 1
    notes = module._annotations()
    case = module.promtool_case(gauges, notes)
    encoded = json.dumps(case)
    for name in module._NAMES:
        assert name in encoded
        assert "docs/incidents/" in notes[name]["runbook"]

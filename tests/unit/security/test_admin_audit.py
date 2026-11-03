"""Admin-key mutations append one JSON line and never log the key."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.admin_audit import (
    admin_context,
    audit_log_env_names,
    cli_identity,
    note_mutation,
)
from quant_fund.api.app import app
from quant_fund.api.http_controls import key_file_env_names
from quant_fund.api.research_api import ResearchApiSettings, create_app
from quant_fund.config.loader import dump_resolved
from quant_fund.config.models import AppConfig
from quant_fund.paper.ledger import PaperLedger

_ADMIN = "audit-admin-key-value"
_READ = "audit-read-key-value"
_FIELDS = {"time", "route", "key_id", "request_id", "outcome", "kind", "prev"}
_WRONG = "wrong-audit-key"


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in key_file_env_names():
        monkeypatch.delenv(name, raising=False)
    for name in (
        "QUANT_API_KEY",
        "QUANT_API_READ_KEY",
        "RESEARCH_API_KEY",
        "RESEARCH_API_READ_KEY",
        *audit_log_env_names(),
    ):
        monkeypatch.delenv(name, raising=False)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_admin_mutation_is_appended_and_the_key_stays_out_of_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text("seed\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.INFO, logger="quant_fund.api.audit"):
        safe = client.get("/models", headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-get"})
        denied = client.post(
            "/backtest",
            headers={"X-API-Key": _READ, "X-Request-ID": _ADMIN},
            json={"config_path": "../pyproject.toml"},
        )
        wrong = client.post(
            "/backtest",
            headers={"X-API-Key": _WRONG, "X-Request-ID": "req-wrong"},
            json={"config_path": "../pyproject.toml"},
        )
        first = client.post(
            "/backtest",
            headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-1"},
            json={"config_path": "../pyproject.toml"},
        )
        second = client.post(
            "/backtest",
            headers={"X-API-Key": _ADMIN, "X-Request-ID": _ADMIN},
            json={"config_path": "../pyproject.toml"},
        )
    assert safe.status_code == 200
    assert denied.status_code == 401
    assert wrong.status_code == 401
    assert first.status_code == 400
    assert second.status_code == 400
    text = log_path.read_text(encoding="utf-8")
    assert text.startswith("seed\n")
    for secret in (_ADMIN, _READ, _WRONG):
        assert secret not in text
        assert secret not in caplog.text
    lines = text.splitlines()
    assert lines[0] == "seed"
    rows = [json.loads(line) for line in lines[1:]]
    assert len(rows) == 4
    assert [row["outcome"] for row in rows] == ["denied", "denied", "400", "400"]
    assert rows[0].keys() == _FIELDS
    assert rows[0]["route"] == "POST /backtest"
    assert rows[0]["kind"] == "ledger"
    assert rows[0]["key_id"] == hashlib.sha256(_READ.encode()).hexdigest()
    assert rows[1]["request_id"] == "req-wrong"
    assert rows[1]["key_id"] == hashlib.sha256(_WRONG.encode()).hexdigest()
    assert rows[2]["request_id"] == "req-1"
    assert rows[2]["key_id"] == hashlib.sha256(_ADMIN.encode()).hexdigest()
    assert rows[2]["time"].endswith("Z")
    assert rows[3]["request_id"] != _ADMIN
    assert rows[3]["key_id"] == rows[2]["key_id"]
    previous = lines[0]
    for line in lines[1:]:
        assert json.loads(line)["prev"] == hashlib.sha256(previous.encode()).hexdigest()
        previous = line
    assert (log_path.stat().st_mode & 0o777) == 0o600


def test_a_closed_audit_log_refuses_the_mutation_before_the_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    blocked = tmp_path / "not-a-file"
    blocked.mkdir()
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(blocked))
    client = TestClient(app, raise_server_exceptions=False)
    safe = client.get("/models", headers={"X-API-Key": _WRONG})
    response = client.post(
        "/backtest",
        headers={"X-API-Key": _ADMIN},
        json={"config_path": "../pyproject.toml"},
    )
    denied = client.post(
        "/backtest",
        headers={"X-API-Key": _WRONG},
        json={"config_path": "../pyproject.toml"},
    )
    assert safe.status_code == 401
    assert response.status_code == 503
    assert denied.status_code == 503
    assert response.json() == {"detail": "audit log unavailable"}
    assert denied.json() == {"detail": "audit log unavailable"}
    assert _ADMIN not in response.text
    assert _WRONG not in denied.text


def test_read_key_mutation_is_not_an_admin_audit_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_READ_KEY", _READ)
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    client = TestClient(app, raise_server_exceptions=False)
    denied = client.post("/portfolio/optimize", headers={"X-API-Key": _READ}, json={})
    assert denied.status_code == 403
    assert denied.json() == {"detail": "read key cannot modify"}
    row = _records(log_path)[0]
    assert row["outcome"] == "denied"
    assert row["kind"] == "optimize"
    assert row["route"] == "POST /portfolio/optimize"
    assert row["key_id"] == hashlib.sha256(_READ.encode()).hexdigest()
    text = log_path.read_text(encoding="utf-8")
    assert _READ not in text
    assert _ADMIN not in text


def test_config_ledger_and_kill_file_notes_keep_the_key_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    with caplog.at_level(logging.INFO, logger="quant_fund.api.audit"):
        note_mutation("config", "written")
        with admin_context("cli:dump_resolved", _ADMIN):
            dump_resolved(AppConfig(), tmp_path / "resolved.json")
        with admin_context("cli:ledger.flush", _ADMIN):
            PaperLedger(tmp_path / "lake", "run1").flush()
        _kill(tmp_path / "desk")
    rows = _records(log_path)
    assert [row["kind"] for row in rows] == ["config", "ledger", "kill-file"]
    assert [row["outcome"] for row in rows] == ["written", "written", "halted"]
    assert [row["route"] for row in rows] == [
        "cli:dump_resolved",
        "cli:ledger.flush",
        "LiveDesk.kill",
    ]
    text = log_path.read_text(encoding="utf-8")
    assert _ADMIN not in text
    assert _ADMIN not in caplog.text
    for row in rows:
        assert row.keys() == _FIELDS
        assert row["key_id"] == hashlib.sha256(_ADMIN.encode()).hexdigest()


def _kill(root: Path) -> None:
    from quant_fund.live.desk import LiveDesk
    from quant_fund.pretrade.config import PretradeConfig, SessionConfig
    from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

    config = PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )
    desk = LiveDesk.open(root, hmac_key=HMAC_KEY, config=config, cash=1_000_000.0, ts_ns=TS_NS)
    try:
        with admin_context("LiveDesk.kill", _ADMIN):
            desk.kill("ops", "halt", TS_NS + 1)
    finally:
        desk.close()


def test_research_admin_post_is_unclassified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    for name in ("data", "receipts", "verifier", "artifacts"):
        (tmp_path / name).mkdir()
    log_path = tmp_path / "research.jsonl"
    monkeypatch.setenv("RESEARCH_API_AUDIT_LOG", str(log_path))
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
        api_key=_ADMIN,
    )
    client = TestClient(create_app(settings), client=("203.0.113.60", 9))
    quiet = client.get("/receipts", headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-get"})
    denied = client.post("/receipts", headers={"X-API-Key": _WRONG, "X-Request-ID": "req-d"})
    response = client.post("/receipts", headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-r"})
    assert quiet.status_code == 200
    assert denied.status_code == 401
    assert response.status_code == 405
    rows = _records(log_path)
    assert rows[0]["outcome"] == "denied"
    assert rows[0]["kind"] == "unclassified"
    assert rows[0]["key_id"] == hashlib.sha256(_WRONG.encode()).hexdigest()
    assert rows[1]["kind"] == "unclassified"
    assert rows[1]["outcome"] == "405"
    assert rows[1]["route"] == "POST /receipts"
    assert rows[1]["request_id"] == "req-r"
    text = log_path.read_text(encoding="utf-8")
    assert _ADMIN not in text
    assert _WRONG not in text


def test_cli_identity_is_user_at_host() -> None:
    import re

    assert re.fullmatch(r"cli:[A-Za-z0-9._-]{1,64}@[A-Za-z0-9._-]{1,64}", cli_identity())


def test_process_mutations_bind_a_cli_actor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    monkeypatch.setattr("quant_fund.api.admin_audit.cli_identity", lambda: "cli:ada@lab")
    dump_resolved(AppConfig(), tmp_path / "resolved.json")
    PaperLedger(tmp_path / "lake", "run1").flush()
    _kill_unbound(tmp_path / "desk")
    _gateway_kill(tmp_path / "gateway")
    rows = _records(log_path)
    assert [row["route"] for row in rows] == ["cli:ada@lab"] * len(rows)
    assert [row["kind"] for row in rows] == ["config", "ledger", "kill-file", "kill-file"]
    assert [row["outcome"] for row in rows] == ["written", "written", "halted", "halted"]
    digest = hashlib.sha256(b"cli:ada@lab").hexdigest()
    assert [row["key_id"] for row in rows] == [digest] * len(rows)
    text = log_path.read_text(encoding="utf-8")
    assert "cli:ada@lab" in text
    assert _ADMIN not in text


def test_portfolio_optimize_kind_is_optimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/portfolio/optimize",
        headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-opt"},
        json={"config_path": "../pyproject.toml"},
    )
    assert response.status_code == 400
    row = _records(log_path)[0]
    assert row["kind"] == "optimize"
    assert row["outcome"] == "400"
    assert row["route"] == "POST /portfolio/optimize"
    assert row["request_id"] == "req-opt"
    assert _ADMIN not in log_path.read_text(encoding="utf-8")


def test_sensitive_reads_append_outcome_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    ledger = "/backtest/" + "ab" * 8
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_READ_KEY", _READ)
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    client = TestClient(app, raise_server_exceptions=False)
    wrong = client.get(ledger, headers={"X-API-Key": _WRONG})
    models = client.get("/models", headers={"X-API-Key": _ADMIN})
    health = client.get("/health")
    doctor = client.get("/doctor", headers={"X-API-Key": _ADMIN})
    read = client.get(
        ledger,
        headers={"X-API-Key": _READ, "X-Request-ID": _ADMIN},
    )
    head = client.head(
        "/backtest/" + "cd" * 8,
        headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-head"},
    )
    positions = client.get(
        "/portfolio/exposures",
        headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-pos"},
    )
    assert wrong.status_code == 401
    assert models.status_code == 200
    assert health.status_code == 200
    assert doctor.status_code == 200
    assert read.status_code == 404
    assert head.status_code < 500
    assert positions.status_code == 200
    text = log_path.read_text(encoding="utf-8")
    for secret in (_ADMIN, _READ, _WRONG):
        assert secret not in text
    rows = _records(log_path)
    assert [row["outcome"] for row in rows] == ["read", "read", "read"]
    assert [row["kind"] for row in rows] == ["ledger", "ledger", "positions"]
    assert rows[0]["route"] == f"GET {ledger}"
    assert rows[0]["key_id"] == hashlib.sha256(_READ.encode()).hexdigest()
    assert rows[0]["request_id"] != _ADMIN
    assert rows[1]["route"] == "HEAD /backtest/" + "cd" * 8
    assert rows[1]["request_id"] == "req-head"
    assert rows[2]["route"] == "GET /portfolio/exposures"
    assert rows[2]["request_id"] == "req-pos"
    assert rows[2]["key_id"] == hashlib.sha256(_ADMIN.encode()).hexdigest()
    monkeypatch.setenv("QUANT_API_AUDIT_READS", "off")
    again = client.get(ledger, headers={"X-API-Key": _ADMIN, "X-Request-ID": "req-off"})
    assert again.status_code == 404
    assert len(_records(log_path)) == 3


def test_loopback_without_a_key_audits_a_sensitive_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    client = TestClient(app, raise_server_exceptions=False)
    quiet = client.get("/models")
    ledger = client.get("/backtest/" + "ab" * 8)
    assert quiet.status_code == 200
    assert ledger.status_code == 404
    rows = _records(log_path)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "read"
    assert rows[0]["kind"] == "ledger"
    assert rows[0]["key_id"] == hashlib.sha256(b"").hexdigest()


def test_a_closed_log_refuses_a_sensitive_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    blocked = tmp_path / "not-a-file"
    blocked.mkdir()
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(blocked))
    client = TestClient(app, raise_server_exceptions=False)
    quiet = client.get("/models", headers={"X-API-Key": _ADMIN})
    response = client.get("/risk/portfolio", headers={"X-API-Key": _ADMIN, "X-Request-ID": _ADMIN})
    assert quiet.status_code == 200
    assert response.status_code == 503
    assert response.json() == {"detail": "audit log unavailable"}
    assert _ADMIN not in response.text


def _kill_unbound(root: Path) -> None:
    from quant_fund.live.desk import LiveDesk
    from quant_fund.pretrade.config import PretradeConfig, SessionConfig
    from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

    config = PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )
    desk = LiveDesk.open(root, hmac_key=HMAC_KEY, config=config, cash=1_000_000.0, ts_ns=TS_NS)
    try:
        desk.kill("ops", "halt", TS_NS + 1)
    finally:
        desk.close()


def _gateway_kill(root: Path) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from quant_fund.live.config import LiveConfig
    from quant_fund.live.session import open_session

    pretrade = Path(__file__).resolve().parents[3] / "configs" / "pretrade_risk.yaml"
    stamp = int(
        datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
    )
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "loopback",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(pretrade),
            "state_dir": str(root),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    session = open_session(config, pretrade, state_dir=root, now_ns=lambda: stamp)
    session.kill(actor="ada", reason="halt", at_ns=stamp)

"""Armed order gateway: pre-trade deny, loopback fill, kill, resume, Alpaca host."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.execution.order_recon import LiveEndpointRefused, refuse_live_endpoint
from quant_fund.live.arming import LIVE_CAPITAL_PHRASE
from quant_fund.live.brokers import AlpacaBroker, LiveOrder, LoopbackBroker
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveGateError
from quant_fund.live.ledger import LedgerFill, LiveLedger
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
SATURDAY = int(
    datetime(2024, 1, 6, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
RUNNER = CliRunner()


def _clock() -> int:
    return TS


def _config(tmp: Path, **overrides: object) -> LiveConfig:
    payload: dict[str, object] = {
        "schema_version": 1,
        "venue": "loopback",
        "starting_cash": 1_000_000.0,
        "calendar": "us_rth",
        "pretrade_config": str(PRETRADE),
        "state_dir": str(tmp),
        "cash_tolerance": 1e-6,
        "position_tolerance": 1e-6,
    }
    payload.update(overrides)
    return LiveConfig.model_validate(payload)


def _open(tmp: Path, **overrides: object) -> Any:
    return open_session(_config(tmp, **overrides), PRETRADE, state_dir=tmp, now_ns=_clock)


def _arm(session: Any, *, phrase: str = "") -> None:
    session.arm(actor="ada", confirm_live_capital=phrase)


def _buy(session: Any, **overrides: object) -> dict[str, Any]:
    payload: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10.0,
        "limit_price": 100.0,
        "mark": 100.0,
        "actor": "ada",
        "at_ns": TS,
    }
    payload.update(overrides)
    return session.submit(**payload)


def test_paper_adapter_still_refuses_a_live_endpoint() -> None:
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint("alpaca-live")


def test_unarmed_submit_does_not_call_the_broker(tmp_path: Path) -> None:
    session = _open(tmp_path)
    with pytest.raises(LiveGateError, match="not armed"):
        _buy(session)
    assert session.broker.account().positions == {}
    assert session.orders == []


def test_loopback_fill_is_accounting_not_a_research_claim(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    result = _buy(session, client_order_id="dc-order-1")
    assert result["ok"] is True
    assert result["status"] == "filled"
    assert result["live_pnl_claim"] is False
    assert result["research_evidence"] is False
    assert result["filled_quantity"] == 10.0
    account = session.broker.account()
    assert account.positions["AAPL"] == 10.0
    assert account.cash == pytest.approx(1_000_000.0 - 1_000.0)
    replay = _buy(session, client_order_id="dc-order-1")
    assert replay["replayed"] is True
    assert session.broker.account().positions["AAPL"] == 10.0
    mode = os.stat(tmp_path / "arm.json").st_mode & 0o777
    assert mode == 0o600


def test_pretrade_collar_blocks_before_the_broker(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    denied = _buy(session, limit_price=110.0)
    assert denied["ok"] is False
    assert denied["broker_called"] is False
    assert "price_collar" in denied["reasons"]
    assert session.broker.account().cash == pytest.approx(1_000_000.0)


def test_kill_switch_blocks_new_orders_and_cancel_still_works(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.broker.defer_fill = True
    working = _buy(session, client_order_id="dc-working")
    assert working["status"] == "accepted"
    session.kill(actor="ada", reason="operator halt", at_ns=TS)
    blocked = _buy(session, quantity=11.0, client_order_id="dc-blocked")
    assert blocked["ok"] is False
    assert "kill_switch" in blocked["reasons"]
    canceled = session.cancel(client_order_id="dc-working", actor="ada", at_ns=TS + 1)
    assert canceled["status"] == "canceled"
    assert session.broker.account().positions == {}


def test_resume_does_not_duplicate_a_fill(tmp_path: Path) -> None:
    first = _open(tmp_path)
    _arm(first)
    _buy(first, client_order_id="dc-resume")
    second = _open(tmp_path)
    replay = _buy(second, client_order_id="dc-resume")
    assert replay["replayed"] is True
    assert second.broker.account().positions["AAPL"] == 10.0
    assert second.broker.account().cash == pytest.approx(1_000_000.0 - 1_000.0)


def test_cash_drift_alarms(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.broker.cash += 50.0
    report = session.reconcile(at_ns=TS + 2)
    assert report["ok"] is False
    assert report["drift"]["alarm"] is True
    assert report["live_pnl_claim"] is False


def test_saturday_rth_is_closed_and_continuous_is_open(tmp_path: Path) -> None:
    closed = open_session(
        _config(tmp_path / "rth"),
        PRETRADE,
        state_dir=tmp_path / "rth",
        now_ns=lambda: SATURDAY,
    )
    _arm(closed)
    denied = closed.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        at_ns=SATURDAY,
    )
    assert "market_closed" in denied["reasons"]

    opened = open_session(
        _config(tmp_path / "all", calendar="continuous"),
        PRETRADE,
        state_dir=tmp_path / "all",
        now_ns=lambda: SATURDAY,
    )
    _arm(opened)
    filled = opened.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        at_ns=SATURDAY,
    )
    assert filled["status"] == "filled"


def test_flatten_sells_the_loopback_position(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session, quantity=4.0)
    flat = session.flatten(actor="ada", at_ns=TS + 5)
    assert flat["ok"] is True
    assert session.broker.account().positions.get("AAPL", 0.0) == pytest.approx(0.0)


def test_alpaca_live_requires_the_phrase_and_refuses_a_backdated_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    session = _open(tmp_path, venue="alpaca-live")
    with pytest.raises(LiveGateError, match="confirm-live-capital"):
        session.arm(actor="ada", confirm_live_capital="please")
    session.arm(actor="ada", confirm_live_capital=LIVE_CAPITAL_PHRASE)
    with pytest.raises(LiveGateError, match="refused for alpaca-live"):
        _buy(session, quantity=1.0)
    with pytest.raises(LiveGateError, match="credentials"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
        )


def test_alpaca_paper_posts_only_to_the_paper_host(tmp_path: Path) -> None:
    transport = _Script()
    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    session = open_session(
        _config(tmp_path, venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    _arm(session)
    result = _buy(session, quantity=1.0, client_order_id="dc-paper-1")
    assert result["status"] == "filled"
    posts = [call for call in transport.calls if call[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1] == "https://paper-api.alpaca.markets/v2/orders"
    assert posts[0][2]["APCA-API-SECRET-KEY"] == "SECRETVALUE"
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    state = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "SECRETVALUE" not in audit
    assert "SECRETVALUE" not in state
    assert "KEYID" not in audit


def test_alpaca_live_host_is_not_the_paper_host() -> None:
    transport = _Script()
    broker = AlpacaBroker("alpaca-live", ("KEYID", "SECRETVALUE"), transport=transport)
    broker.submit(
        LiveOrder(
            client_order_id="dc-live-1",
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            limit_price=100.0,
            fill_time_ns=TS,
        )
    )
    assert transport.calls[0][1].startswith("https://api.alpaca.markets/")


def test_ledger_duplicate_orphan_and_out_of_order_are_idempotent() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("a", expects_fill=True)
    first = LedgerFill("f1", "a", "AAPL", "buy", 1.0, 10.0, 20)
    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(first) == "duplicate_fill"
    assert ledger.computed_cash == pytest.approx(990.0)
    late = LedgerFill("f0", "a", "AAPL", "buy", 1.0, 10.0, 10)
    assert ledger.record_fill(late) == "out_of_order_fill"
    orphan = LedgerFill("fx", "missing", "AAPL", "buy", 1.0, 10.0, 30)
    assert ledger.record_fill(orphan) == "orphan_fill"
    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["orphan_fills"] == 1
    assert report["out_of_order_fills"] == 1
    assert report["live_pnl_claim"] is False


def test_state_that_claims_live_profit_is_refused(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session)
    path = tmp_path / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["live_pnl_claim"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LiveGateError, match="live profit"):
        _open(tmp_path)


def test_cli_arm_and_submit(tmp_path: Path) -> None:
    config_path = tmp_path / "live.yaml"
    state = tmp_path / "state"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "venue": "loopback",
                "starting_cash": 100000,
                "calendar": "us_rth",
                "pretrade_config": str(PRETRADE),
                "state_dir": str(state),
                "cash_tolerance": 1e-6,
                "position_tolerance": 1e-6,
            }
        ),
        encoding="utf-8",
    )
    common = ["--config", str(config_path), "--state-dir", str(state)]
    unarmed = RUNNER.invoke(
        app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
            "--actor",
            "ada",
            "--at",
            "2024-01-03T15:00:00-05:00",
        ],
    )
    assert unarmed.exit_code == 1
    armed = RUNNER.invoke(app, ["live", "arm", *common, "--actor", "ada"])
    assert armed.exit_code == 0
    submitted = RUNNER.invoke(
        app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
            "--actor",
            "ada",
            "--at",
            "2024-01-03T15:00:00-05:00",
        ],
    )
    assert submitted.exit_code == 0, submitted.stdout + submitted.stderr
    body = json.loads(submitted.stdout)
    assert body["status"] == "filled"
    assert body["live_pnl_claim"] is False
    status = RUNNER.invoke(app, ["live", "status", *common])
    assert status.exit_code == 0
    snapshot = json.loads(status.stdout)
    assert snapshot["armed"] is True
    assert snapshot["computed_positions"]["AAPL"] == 1.0


class _Script:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.filled = False

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del timeout
        self.calls.append((method, url, headers, body))
        if method == "POST":
            self.filled = True
            request = json.loads(body or b"{}")
            payload = {
                "id": "ord-1",
                "client_order_id": request["client_order_id"],
                "status": "filled",
                "filled_qty": request["qty"],
                "filled_avg_price": request["limit_price"],
                "symbol": request["symbol"],
                "side": request["side"],
                "filled_at": "2024-01-03T20:00:00+00:00",
            }
            return 200, json.dumps(payload).encode()
        if url.endswith("/v2/account"):
            cash = "999900" if self.filled else "1000000"
            return 200, json.dumps({"cash": cash, "equity": "1000000"}).encode()
        if url.endswith("/v2/positions"):
            rows = [{"symbol": "AAPL", "qty": "1"}] if self.filled else []
            return 200, json.dumps(rows).encode()
        if "activities" in url:
            return 200, b"[]"
        return 404, b"{}"

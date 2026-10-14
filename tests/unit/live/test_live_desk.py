"""Operator order desk. No network. SYNTHETIC broker payloads are fixtures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pytest
from tests.unit.pretrade.support import HMAC_KEY, limit_config
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.live.alpaca import AlpacaClient
from quant_fund.live.authorization import (
    ACK_PHRASE,
    LIVE_CONFIRM_ENV,
    LIVE_CONFIRM_VALUE,
    LiveAuthorization,
    assert_live_confirmed,
)
from quant_fund.live.desk import LiveDesk, open_desk, preflight_report
from quant_fund.live.errors import BrokerUncertain, LiveRefused, TransportError
from quant_fund.live.transport import HttpResponse, UrllibTransport
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.pretrade.shadow import business_session_id
from quant_fund.schemas.errors import KillSwitchActive

NOW = datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York"))
CLOSED = datetime(2024, 1, 6, 15, 0, tzinfo=ZoneInfo("America/New_York"))
KEY = "example-key-id"
SECRET = "example-secret"


def _auth(**overrides: object) -> LiveAuthorization:
    payload: dict[str, object] = {
        "schema": "live_authorization.v1",
        "environment": "paper",
        "broker": "alpaca",
        "account_id": "PA123",
        "operator": "tester",
        "acknowledge": ACK_PHRASE,
        "symbol_allowlist": ["AAPL"],
        "max_order_notional": 50_000.0,
        "max_orders_per_day": 5,
        "created_at": "2024-01-01T00:00:00+00:00",
        "expires_at": "2024-03-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return LiveAuthorization.model_validate(payload)


def _config(**overrides: object) -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(**overrides),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


class ScriptedTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.routes: dict[tuple[str, str], tuple[int, dict[str, Any] | list[Any]]] = {}
        self.fail_post = False

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HttpResponse:
        assert headers["APCA-API-KEY-ID"] == KEY
        assert SECRET not in url
        self.calls.append((method, url, body))
        if method == "POST" and self.fail_post:
            raise TransportError("down")
        status, payload = self.routes[(method, urlsplit(url).path)]
        return HttpResponse(status=status, body=json.dumps(payload).encode())


def _routes(
    transport: ScriptedTransport,
    *,
    equity: str = "100000",
    positions: list[dict[str, str]] | None = None,
    post_status: int = 200,
) -> None:
    transport.routes[("GET", "/v2/account")] = (
        200,
        {
            "id": "id-1",
            "account_number": "PA123",
            "status": "ACTIVE",
            "equity": equity,
            "cash": equity,
        },
    )
    transport.routes[("GET", "/v2/positions")] = (200, positions or [])
    transport.routes[("GET", "/v2/stocks/AAPL/trades/latest")] = (200, {"trade": {"p": 100}})
    transport.routes[("GET", "/v2/stocks/MSFT/trades/latest")] = (200, {"trade": {"p": 100}})
    transport.routes[("POST", "/v2/orders")] = (
        post_status,
        {
            "id": "ord-1",
            "client_order_id": "ignored",
            "status": "accepted",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "1",
            "message": "no",
        },
    )
    transport.routes[("DELETE", "/v2/orders/ord-1")] = (204, {})
    transport.routes[("DELETE", "/v2/orders")] = (207, [])
    transport.routes[("GET", "/v2/orders:by_client_order_id")] = (404, {"message": "missing"})


def _desk(
    tmp_path: Path,
    transport: ScriptedTransport,
    *,
    clock: datetime = NOW,
    auth_overrides: dict[str, object] | None = None,
    limit_overrides: dict[str, object] | None = None,
    arm: bool = True,
) -> LiveDesk:
    desk = LiveDesk(
        authorization=_auth(**(auth_overrides or {})),
        authorization_sha256="sha",
        config=_config(**(limit_overrides or {})),
        hmac_key=HMAC_KEY,
        broker=AlpacaClient(environment="paper", key_id=KEY, secret=SECRET, transport=transport),
        state_dir=tmp_path,
        clock=lambda: clock,
    )
    if arm:
        desk.arm()
    return desk


def _posts(transport: ScriptedTransport) -> list[tuple[str, str, bytes | None]]:
    return [call for call in transport.calls if call[0] == "POST"]


def test_limit_order_routes_to_paper_and_stays_honest(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    result = desk.submit(symbol="aapl", side="buy", qty=1, limit_price=100)
    assert result["accepted"] is True
    assert result["live_pnl_claim"] is False
    assert result["would_promote_live"] is False
    assert result["institutional_live_claim"] is False
    assert result["evidence_class"] == "broker_routing"
    posts = _posts(transport)
    assert len(posts) == 1
    assert posts[0][1].startswith("https://paper-api.alpaca.markets/v2/orders")
    body = json.loads(posts[0][2] or b"{}")
    assert body["type"] == "limit"
    assert body["extended_hours"] is False
    assert body["symbol"] == "AAPL"
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert SECRET not in audit
    assert KEY not in audit


def test_unarmed_desk_does_not_call_the_broker(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport, arm=False)
    with pytest.raises(LiveRefused, match="not armed"):
        desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert transport.calls == []


def test_allowlist_and_notional_deny_before_any_request(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    with pytest.raises(LiveRefused, match="allowlist"):
        desk.submit(symbol="MSFT", side="buy", qty=1, limit_price=100)
    denied = desk.submit(symbol="AAPL", side="buy", qty=10, limit_price=10_000)
    assert denied["accepted"] is False
    assert denied["reasons"] == ["max_order_notional"]
    assert _posts(transport) == []


def test_kill_switch_blocks_new_orders(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    desk.kill(actor="op", reason="halt")
    with pytest.raises(KillSwitchActive):
        desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert _posts(transport) == []
    status = desk.status()
    assert status["kill_latched"] is True
    assert status["live_pnl_claim"] is False


def test_broker_reject_does_not_latch_kill(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport, post_status=422)
    desk = _desk(tmp_path, transport)
    result = desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert result["accepted"] is False
    assert result["broker_rejected"] is True
    assert desk.kill_latched() is False


def test_uncertain_submit_latches_kill(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    transport.fail_post = True
    desk = _desk(tmp_path, transport)
    with pytest.raises(BrokerUncertain):
        desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert desk.kill_latched() is True
    with pytest.raises(KillSwitchActive):
        desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=101)


def test_server_error_on_submit_is_uncertain(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport, post_status=500)
    desk = _desk(tmp_path, transport)
    with pytest.raises(BrokerUncertain):
        desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert desk.kill_latched() is True


def test_duplicate_order_is_not_sent_again(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    assert desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)["accepted"] is True
    duplicate = desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert duplicate["reasons"] == ["duplicate_order"]
    assert len(_posts(transport)) == 1


def test_daily_order_cap(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport, auth_overrides={"max_orders_per_day": 1})
    assert desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)["accepted"] is True
    capped = desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=101)
    assert capped["reasons"] == ["order_rate"]
    assert len(_posts(transport)) == 1


def test_client_order_id_is_idempotent(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    client_id = "client-order-0001"
    first = desk.submit(
        symbol="AAPL", side="buy", qty=1, limit_price=100, client_order_id=client_id
    )
    desk.kill(actor="op", reason="halt")
    second = desk.submit(
        symbol="AAPL", side="buy", qty=1, limit_price=100, client_order_id=client_id
    )
    assert second["idempotent"] is True
    assert second["broker_order_id"] == first["broker_order_id"]
    assert len(_posts(transport)) == 1


def test_closed_market_does_not_post(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport, clock=CLOSED)
    result = desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert result["accepted"] is False
    assert "market_closed" in result["reasons"]
    assert _posts(transport) == []


def test_session_loss_latches_and_blocks(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport, equity="97000")
    desk = _desk(tmp_path, transport, limit_overrides={"max_daily_loss_fraction": 0.02})
    (tmp_path / "session.json").write_text(
        json.dumps(
            {"day": "2024-01-03", "session_start_equity": 100000.0, "peak_equity": 100000.0}
        ),
        encoding="utf-8",
    )
    result = desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    assert result["accepted"] is False
    assert "daily_loss" in result["reasons"]
    assert desk.kill_latched() is True
    assert _posts(transport) == []


def test_flatten_reduce_only_while_killed(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(
        transport,
        positions=[{"symbol": "AAPL", "qty": "2", "current_price": "100"}],
    )
    desk = _desk(tmp_path, transport)
    desk.kill(actor="op", reason="halt")
    result = desk.flatten(confirm="FLATTEN")
    assert result["kill_latched"] is True
    assert result["results"][0]["accepted"] is True
    body = json.loads(_posts(transport)[0][2] or b"{}")
    assert body["side"] == "sell"
    assert body["type"] == "market"
    assert body["qty"] == "2"
    with pytest.raises(LiveRefused, match="FLATTEN"):
        desk.flatten(confirm="no")


def test_account_mismatch_and_inactive_account(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    transport.routes[("GET", "/v2/account")] = (
        200,
        {
            "id": "other",
            "account_number": "OTHER",
            "status": "ACTIVE",
            "equity": "100000",
            "cash": "100000",
        },
    )
    desk = _desk(tmp_path, transport)
    with pytest.raises(LiveRefused, match="does not match"):
        desk.submit(symbol="AAPL", side="buy", qty=1, limit_price=100)
    transport.routes[("GET", "/v2/account")] = (
        200,
        {
            "id": "id-1",
            "account_number": "PA123",
            "status": "ACCOUNT_CLOSED",
            "equity": "100000",
            "cash": "100000",
        },
    )
    with pytest.raises(LiveRefused, match="ACCOUNT_CLOSED"):
        desk.sync()
    assert _posts(transport) == []


def test_reconcile_reports_a_missing_broker_order(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    submitted = desk.submit(
        symbol="AAPL", side="buy", qty=1, limit_price=100, client_order_id="client-order-0002"
    )
    report = desk.reconcile()
    assert report["ok"] is False
    assert report["missing_at_broker"] == [submitted["client_order_id"]]
    assert report["live_pnl_claim"] is False


def test_cancel_and_cancel_all(tmp_path: Path) -> None:
    transport = ScriptedTransport()
    _routes(transport)
    desk = _desk(tmp_path, transport)
    assert desk.cancel("ord-1")["accepted"] is True
    assert desk.cancel_all()["accepted"] is True
    assert any(call[0] == "DELETE" for call in transport.calls)


def test_live_environment_uses_the_live_host_and_needs_confirmation() -> None:
    transport = ScriptedTransport()
    _routes(transport)
    client = AlpacaClient(environment="live", key_id=KEY, secret=SECRET, transport=transport)
    with pytest.raises(LiveRefused, match=LIVE_CONFIRM_ENV):
        assert_live_confirmed("live", environ={})
    assert_live_confirmed("live", environ={LIVE_CONFIRM_ENV: LIVE_CONFIRM_VALUE})
    # The client builds the live origin. The script records the URL and then
    # fails closed on the missing route for this symbol-less call shape.
    transport.routes[("POST", "/v2/orders")] = (
        200,
        {
            "id": "ord-9",
            "client_order_id": "client-order-0003",
            "status": "accepted",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "1",
        },
    )
    client.submit_order(
        symbol="AAPL", side="buy", qty=1, limit_price=10, client_order_id="client-order-0003"
    )
    assert transport.calls[-1][1].startswith("https://api.alpaca.markets/")


def test_transport_refuses_other_hosts() -> None:
    transport = UrllibTransport(environment="paper")
    with pytest.raises(LiveRefused, match="not allowed"):
        transport.request(
            method="GET",
            url="https://evil.example/v2/account",
            headers={},
            body=None,
            timeout_s=1,
        )
    with pytest.raises(LiveRefused, match="https"):
        transport.request(
            method="GET",
            url="http://paper-api.alpaca.markets/v2/account",
            headers={},
            body=None,
            timeout_s=1,
        )


def test_authorization_rejects_a_bad_phrase_and_a_long_life() -> None:
    with pytest.raises(ValueError, match="acknowledge"):
        _auth(acknowledge="I agree")
    with pytest.raises(ValueError, match="90 days"):
        _auth(expires_at="2024-06-01T00:00:00+00:00")


def test_session_index_matches_the_pretrade_rule() -> None:
    from quant_fund.live.desk import _business_session_id

    assert _business_session_id(NOW.date()) == business_session_id(NOW.date())


def test_harness_cannot_call_live() -> None:
    from fx1.harness import HARNESS_REGISTRY

    assert "live" not in {command.name for command in HARNESS_REGISTRY}


def test_example_authorization_file_matches_the_schema() -> None:
    path = Path("configs/live_authorization.example.json")
    LiveAuthorization.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_open_desk_and_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "schema": "live_authorization.v1",
                "environment": "paper",
                "broker": "alpaca",
                "account_id": "PA123",
                "operator": "tester",
                "acknowledge": ACK_PHRASE,
                "symbol_allowlist": ["AAPL"],
                "max_order_notional": 1000,
                "max_orders_per_day": 2,
                "created_at": "2024-01-01T00:00:00+00:00",
                "expires_at": "2024-03-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DIPCATCHER_BROKER_KEY", KEY)
    monkeypatch.setenv("DIPCATCHER_BROKER_SECRET", SECRET)
    monkeypatch.setenv("DIPCATCHER_PRETRADE_HMAC_KEY", HMAC_KEY.hex())
    transport = ScriptedTransport()
    _routes(transport)
    desk = open_desk(
        authorization=auth_path,
        pretrade=Path("configs/pretrade_risk.yaml"),
        state_dir=tmp_path / "state",
        transport=transport,
        clock=lambda: NOW,
    )
    assert desk.status()["armed"] is False
    report = preflight_report(
        authorization_path=auth_path,
        pretrade_path=Path("configs/pretrade_risk.yaml"),
        state_dir=tmp_path / "state",
        now=NOW,
    )
    assert report["ok"] is True
    assert report["broker_key_present"] is True
    assert SECRET not in json.dumps(report)


def test_cli_order_deny_and_help(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Fake:
        def submit(self, **_kwargs: object) -> dict[str, object]:
            return {
                "accepted": False,
                "reasons": ["max_order_notional"],
                "institutional_live_claim": False,
                "would_promote_live": False,
                "live_pnl_claim": False,
            }

    monkeypatch.setattr("quant_fund.cli.live_cmds.open_desk", lambda **_kwargs: _Fake())
    runner = CliRunner()
    denied = runner.invoke(
        app,
        ["live", "order", "--symbol", "AAPL", "--side", "buy", "--qty", "1", "--limit-price", "10"],
    )
    assert denied.exit_code == 3
    assert "max_order_notional" in denied.stdout
    help_result = runner.invoke(app, ["live", "--help"])
    assert help_result.exit_code == 0
    assert "arm" in help_result.stdout
    missing = runner.invoke(
        app, ["live", "order", "--symbol", "AAPL", "--side", "buy", "--qty", "1"]
    )
    assert missing.exit_code != 0

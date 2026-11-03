from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import AlpacaBroker
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveGateError
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


class _Transport:
    def __init__(self, post_response: tuple[int, bytes], *, recover_missing: bool = False) -> None:
        self.post_response = post_response
        self.recover_missing = recover_missing
        self.post_calls = 0
        self.recover_calls = 0

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, body, timeout
        if url.endswith("/v2/account"):
            return 200, json.dumps({"cash": "1000000", "equity": "1000000"}).encode()
        if url.endswith("/v2/positions"):
            return 200, b"[]"
        if method == "POST" and url.endswith("/v2/orders"):
            self.post_calls += 1
            return self.post_response
        if method == "GET" and "/v2/orders:by_client_order_id?" in url:
            self.recover_calls += 1
            if self.recover_missing:
                return 404, b""
            return 200, json.dumps(
                {
                    "id": "broker-order",
                    "client_order_id": "dc-ambiguous",
                    "status": "accepted",
                    "filled_qty": "0",
                    "symbol": "AAPL",
                    "side": "buy",
                }
            ).encode()
        raise AssertionError(f"unexpected request: {method} {url}")


def _submit(tmp_path: Path, transport: _Transport) -> dict[str, Any]:
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "alpaca-paper",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp_path),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=transport)
    session = open_session(config, PRETRADE, state_dir=tmp_path, now_ns=lambda: TS, broker=broker)
    session.arm(actor="ada")
    return session.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        client_order_id="dc-ambiguous",
        at_ns=TS,
    )


@pytest.mark.parametrize(
    "response",
    [
        (408, b'{"message":"timeout"}'),
        (429, b'{"message":"rate limited"}'),
        (503, b'{"message":"unavailable"}'),
        (200, b"not-json"),
        (200, b"[]"),
    ],
    ids=["http-408", "http-429", "http-503", "non-json-200", "non-object-200"],
)
def test_uncertain_post_recovers_instead_of_rejecting(
    tmp_path: Path, response: tuple[int, bytes]
) -> None:
    transport = _Transport(response)

    result = _submit(tmp_path, transport)

    assert result["ok"] is True
    assert result["status"] == "accepted"
    assert transport.post_calls == 1
    assert transport.recover_calls == 1


def test_unrecovered_uncertain_post_stays_inflight(tmp_path: Path) -> None:
    transport = _Transport((503, b'{"message":"unavailable"}'), recover_missing=True)

    with pytest.raises(LiveGateError, match="stays inflight"):
        _submit(tmp_path, transport)

    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "inflight"
    assert transport.post_calls == 1
    assert transport.recover_calls == 1


def test_definitive_broker_reject_does_not_recover(tmp_path: Path) -> None:
    transport = _Transport((422, b'{"message":"invalid order"}'))

    result = _submit(tmp_path, transport)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert transport.post_calls == 1
    assert transport.recover_calls == 0

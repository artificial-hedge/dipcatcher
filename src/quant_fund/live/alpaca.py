"""Alpaca trading client.

Paper and live are different origins. A paper client cannot construct a live
trading URL. This module does not decide whether the operator is allowed to
trade; ``quant_fund.live.desk`` does.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from quant_fund.live.errors import BrokerRejected, BrokerUncertain, TransportError
from quant_fund.live.transport import (
    DATA_BASE,
    HttpTransport,
    redact,
    response_message,
    trading_base,
)

_MUTATING = frozenset({"POST", "DELETE"})


@dataclass(frozen=True)
class AccountState:
    account_id: str
    account_number: str
    status: str
    equity: float
    cash: float


@dataclass(frozen=True)
class PositionState:
    symbol: str
    qty: float
    price: float


@dataclass(frozen=True)
class OrderAck:
    broker_order_id: str
    client_order_id: str
    status: str
    symbol: str
    side: str
    qty: str


def _number(value: object, label: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BrokerRejected(f"{label} is missing", status=0)
    try:
        number = float(value)
    except ValueError as exc:
        raise BrokerRejected(f"{label} is not a number", status=0) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise BrokerRejected(f"{label} is not a usable number", status=0)
    return number


def _qty_text(qty: float) -> str:
    text = format(qty, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


class AlpacaClient:
    """Authenticated Alpaca REST calls through an injected transport."""

    def __init__(
        self,
        *,
        environment: str,
        key_id: str,
        secret: str,
        transport: HttpTransport,
        timeout_s: float = 10.0,
    ) -> None:
        self.environment = environment
        self._base = trading_base(environment)
        self._key_id = key_id
        self._secret = secret
        self._transport = transport
        self._timeout_s = float(timeout_s)

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key_id,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }

    def _call(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        *,
        mutating: bool | None = None,
    ) -> Any:
        is_mutating = method in _MUTATING if mutating is None else mutating
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            response = self._transport.request(
                method=method,
                url=url,
                headers=headers,
                body=body,
                timeout_s=self._timeout_s,
            )
        except TransportError as exc:
            if is_mutating:
                raise BrokerUncertain("broker request did not complete") from exc
            raise BrokerRejected("broker read did not complete", status=0) from exc
        return self._decode(response.status, response.body, mutating=is_mutating)

    def _decode(self, status: int, body: bytes, *, mutating: bool) -> Any:
        message = response_message(body, self._key_id, self._secret)
        if status >= 500 or status == 0:
            if mutating:
                raise BrokerUncertain(f"broker status {status}: {message}", status=status)
            raise BrokerRejected(f"broker status {status}: {message}", status=status)
        if status >= 400:
            raise BrokerRejected(f"broker status {status}: {message}", status=status)
        if not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            if mutating:
                raise BrokerUncertain("broker returned a non-JSON mutation response") from exc
            raise BrokerRejected("broker returned non-JSON", status=status) from exc

    def get_account(self) -> AccountState:
        payload = self._call("GET", f"{self._base}/v2/account")
        if not isinstance(payload, dict):
            raise BrokerRejected("account response is not an object", status=0)
        return AccountState(
            account_id=str(payload.get("id", "")),
            account_number=str(payload.get("account_number", "")),
            status=str(payload.get("status", "")),
            equity=_number(payload.get("equity"), "equity", positive=True),
            cash=_number(payload.get("cash"), "cash", positive=False),
        )

    def get_positions(self) -> tuple[PositionState, ...]:
        payload = self._call("GET", f"{self._base}/v2/positions")
        if not isinstance(payload, list):
            raise BrokerRejected("positions response is not a list", status=0)
        rows: list[PositionState] = []
        for item in payload:
            if not isinstance(item, dict):
                raise BrokerRejected("position row is not an object", status=0)
            price = item.get("current_price", item.get("avg_entry_price"))
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                raise BrokerRejected("position is missing a symbol", status=0)
            rows.append(
                PositionState(
                    symbol=symbol,
                    qty=_number(item.get("qty"), "position qty", positive=False),
                    price=_number(price, "position price", positive=True),
                )
            )
        return tuple(rows)

    def latest_trade(self, symbol: str) -> float:
        safe = quote(symbol, safe="")
        payload = self._call("GET", f"{DATA_BASE}/v2/stocks/{safe}/trades/latest")
        if not isinstance(payload, dict) or not isinstance(payload.get("trade"), dict):
            raise BrokerRejected("trade response has no trade", status=0)
        trade = payload["trade"]
        return _number(trade.get("p"), "reference price", positive=True)

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float | None,
        client_order_id: str,
    ) -> OrderAck:
        order: dict[str, object] = {
            "symbol": symbol,
            "qty": _qty_text(qty),
            "side": side,
            "time_in_force": "day",
            "client_order_id": client_order_id,
            "extended_hours": False,
        }
        if limit_price is None:
            order["type"] = "market"
        else:
            order["type"] = "limit"
            order["limit_price"] = _qty_text(limit_price)
        body = json.dumps(order, separators=(",", ":")).encode("utf-8")
        payload = self._call("POST", f"{self._base}/v2/orders", body)
        return _ack(payload)

    def get_order_by_client_id(self, client_order_id: str) -> OrderAck:
        query = urlencode({"client_order_id": client_order_id})
        payload = self._call("GET", f"{self._base}/v2/orders:by_client_order_id?{query}")
        return _ack(payload)

    def cancel_order(self, broker_order_id: str) -> None:
        safe = quote(broker_order_id, safe="")
        self._call("DELETE", f"{self._base}/v2/orders/{safe}")

    def cancel_all(self) -> None:
        self._call("DELETE", f"{self._base}/v2/orders")

    def redact_text(self, text: str) -> str:
        """Strip stored credentials from a record before it is written."""
        return redact(text, self._key_id, self._secret)


def _ack(payload: object) -> OrderAck:
    if not isinstance(payload, Mapping):
        raise BrokerRejected("order response is not an object", status=0)
    return OrderAck(
        broker_order_id=str(payload.get("id", "")),
        client_order_id=str(payload.get("client_order_id", "")),
        status=str(payload.get("status", "")),
        symbol=str(payload.get("symbol", "")).upper(),
        side=str(payload.get("side", "")),
        qty=str(payload.get("qty", "")),
    )

"""Fault-injection fakes. SYNTHETIC correctness fixtures, never a live broker.

No transport, no alpaca-live host, and no real-broker credentials.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

from quant_fund.live.brokers import BrokerAccount, BrokerAck, BrokerFill, FillPage, LiveOrder
from quant_fund.live.config import LiveConfig
from quant_fund.live.desk import LiveDesk
from quant_fund.live.errors import BrokerRejected
from quant_fund.live.session import open_session
from quant_fund.pretrade.config import PretradeConfig, SessionConfig

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
SEEDS = (1, 2, 3, 7, 11, 19, 23, 42)
STARTING_CASH = 1_000_000.0


class Scheduler:
    """Seeded shuffle. The same seed replays the same interleaving."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.trace: list[tuple[str, ...]] = []

    def interleave(self, events: list[str]) -> tuple[str, ...]:
        row = list(events)
        self._rng.shuffle(row)
        done = tuple(row)
        self.trace.append(done)
        return done


class FaultBroker:
    """In-memory paper venue. Idempotent submit, droppable ack, no sockets."""

    def __init__(self, *, cash: float = STARTING_CASH) -> None:
        self.cash = float(cash)
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.orders: dict[str, LiveOrder] = {}
        self._acks: dict[str, BrokerAck] = {}
        self._page: list[BrokerFill] = []
        self._applied: set[str] = set()
        self._filled: dict[str, float] = {}
        self.submit_count = 0
        self.connected = True
        self.drop_ack = False
        self.drop_recover = False

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        equity = self.cash
        for symbol, qty in self.positions.items():
            equity += qty * self.marks.get(symbol, 0.0)
        return BrokerAccount(
            cash=self.cash,
            equity=equity,
            positions=dict(self.positions),
            marks=dict(self.marks),
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        if order.client_order_id not in self._acks:
            self.submit_count += 1
            self.orders[order.client_order_id] = order
            self._filled[order.client_order_id] = 0.0
            self._acks[order.client_order_id] = BrokerAck(
                client_order_id=order.client_order_id,
                broker_order_id=f"paper:{order.client_order_id}",
                status="accepted",
                filled_quantity=0.0,
                fill_price=None,
                fills=(),
                reason="",
            )
        if self.drop_ack or not self.connected:
            raise TimeoutError("broker ack dropped")
        return self._acks[order.client_order_id]

    def recover(self, client_order_id: str) -> BrokerAck | None:
        if self.drop_recover or not self.connected:
            raise TimeoutError("broker recover dropped")
        return self._acks.get(client_order_id)

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del broker_order_id
        ack = self._acks.get(client_order_id)
        if ack is None:
            raise BrokerRejected("order is not at the broker")
        if ack.status in {"canceled", "filled", "rejected"}:
            return ack
        updated = BrokerAck(
            client_order_id=client_order_id,
            broker_order_id=ack.broker_order_id,
            status="canceled",
            filled_quantity=ack.filled_quantity,
            fill_price=ack.fill_price,
            fills=self._applied_fills(client_order_id),
            reason="",
        )
        self._acks[client_order_id] = updated
        return updated

    def deliver(
        self,
        client_order_id: str,
        qty: float,
        price: float,
        ts: int,
        fill_id: str,
    ) -> str:
        """Apply a fill once. A fill after cancel, or past the order qty, is visible only."""
        if any(item.fill_id == fill_id for item in self._page):
            return "duplicate"
        order = self.orders.get(client_order_id)
        fill = BrokerFill(
            fill_id=fill_id,
            client_order_id=client_order_id,
            symbol=order.symbol if order is not None else "AAPL",
            side=order.side if order is not None else "buy",
            quantity=qty,
            price=price,
            fill_time_ns=ts,
        )
        self._page.append(fill)
        if order is None:
            return "unknown"
        ack = self._acks[client_order_id]
        remaining = order.quantity - self._filled.get(client_order_id, 0.0)
        if ack.status in {"canceled", "rejected", "filled"} or qty > remaining + 1e-9:
            return "unapplied"
        signed = qty if order.side == "buy" else -qty
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + signed
        self.cash -= signed * price
        self._filled[client_order_id] = self._filled.get(client_order_id, 0.0) + qty
        self._applied.add(fill_id)
        filled = self._filled[client_order_id]
        status = "filled" if filled + 1e-9 >= order.quantity else "partially_filled"
        self._acks[client_order_id] = BrokerAck(
            client_order_id=client_order_id,
            broker_order_id=ack.broker_order_id,
            status=status,
            filled_quantity=filled,
            fill_price=price,
            fills=self._applied_fills(client_order_id),
            reason="",
        )
        return "applied"

    def inject_unknown(self, *, fill_id: str, quantity: float, price: float, ts: int) -> None:
        self._page.append(
            BrokerFill(
                fill_id=fill_id,
                client_order_id="UNKNOWN-ORDER",
                symbol="AAPL",
                side="buy",
                quantity=quantity,
                price=price,
                fill_time_ns=ts,
            )
        )

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=tuple(self._page), truncated=False)

    def _applied_fills(self, client_order_id: str) -> tuple[BrokerFill, ...]:
        return tuple(
            item
            for item in self._page
            if item.client_order_id == client_order_id and item.fill_id in self._applied
        )


def clock_at(stamp: int) -> Callable[[], int]:
    return lambda: stamp


def open_fault_session(tmp: Path, broker: object, *, venue: str = "alpaca-paper", now: int = TS):
    if venue == "alpaca-live":
        raise AssertionError("fault injection must not open alpaca-live")
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": venue,
            "starting_cash": STARTING_CASH,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    return open_session(config, PRETRADE, state_dir=tmp, now_ns=clock_at(now), broker=broker)


def arm(session: object) -> None:
    session.arm(actor="ada")  # type: ignore[attr-defined]


def buy(session: object, **overrides: object) -> dict[str, object]:
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
    return session.submit(**payload)  # type: ignore[attr-defined, no-any-return]


def open_desk(tmp: Path, **overrides: object) -> LiveDesk:
    root = tmp / "desk"
    funded = not (root / "state.json").exists()
    cash = STARTING_CASH if funded else None
    return LiveDesk.open(
        root,
        hmac_key=HMAC_KEY,
        config=PretradeConfig(
            schema_version=1,
            limits=limit_config(**overrides),
            session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
        ),
        cash=cash,
        account_value=cash,
        ts_ns=TS_NS,
    )


def first_effect(names: tuple[str, ...]) -> str:
    """First fill-or-cancel in a seeded trace. A duplicate fill id is still a fill."""
    for name in names:
        if name == "cancel":
            return "cancel"
        if name in {"fill", "dup"}:
            return "fill"
    return "fill"

"""Retries and duplicate client order ids apply an order once.

The paper ledger and the simulated broker key on ``order_id``. Loopback and
the desk key on ``client_order_id``. A retry returns the first acknowledgement,
including when it names a different quantity or side. Fills are SYNTHETIC and
``live_pnl_claim`` stays false.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.live.brokers import LiveOrder, LoopbackBroker
from quant_fund.live.desk import LiveDesk
from quant_fund.paper.broker_adapter import PaperBrokerAdapter
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.schemas.orders import Order as BrokerOrder
from quant_fund.schemas.orders import OrderSide, OrderStatus
from tests.property._books import research_config
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

_FAST = settings(derandomize=True, max_examples=20, deadline=None)
_T0 = datetime(2024, 1, 2, tzinfo=UTC)
_CASH = 100_000.0
_CLIENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12)


def _desk_config() -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _open_desk(root: Path) -> LiveDesk:
    return LiveDesk.open(
        root / "desk",
        hmac_key=HMAC_KEY,
        config=_desk_config(),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def _order(order_id: str, side: str, qty: int) -> Order:
    return Order(order_id=order_id, security_id="A", side=side, quantity=float(qty))


def _fill(fill_id: str, qty: int, price: int, *, seconds: int) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id="o1",
        security_id="A",
        quantity=float(qty),
        price=float(price),
        fill_time=_T0 + timedelta(seconds=seconds),
    )


@given(
    qty=st.integers(1, 12),
    extra=st.integers(1, 6),
    price=st.integers(10, 40),
    retries=st.integers(1, 4),
    side=st.sampled_from(("BUY", "SELL")),
)
@_FAST
def test_duplicate_order_id_keeps_the_first_order(
    qty: int, extra: int, price: int, retries: int, side: str
) -> None:
    other = "SELL" if side == "BUY" else "BUY"
    ledger = OrderFillLedger(endpoint_kind="simulated", starting_cash=_CASH)
    assert ledger.submit(_order("o1", side, qty)) == "accepted"
    for _ in range(retries):
        assert ledger.submit(_order("o1", other, qty + extra)) == "duplicate_order"
    assert ledger.record_fill(_fill("f1", qty, price, seconds=1)) == "accepted"
    assert ledger.record_fill(_fill("f1", qty, price, seconds=1)) == "duplicate_fill"
    assert ledger.record_fill(_fill("f2", extra, price, seconds=2)) == "overfill"
    report = ledger.reconcile()
    assert report["n_orders"] == 1
    assert report["counters"]["duplicate_orders"] == retries
    assert report["counters"]["duplicate_fills"] == 1
    assert report["counters"]["orders"] == 1
    signed = float(qty if side == "BUY" else -qty)
    assert report["computed_positions"]["A"] == pytest.approx(signed)
    assert report["computed_cash"] == pytest.approx(_CASH - signed * price)
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    row = report["orders"][0]
    assert row["quantity"] == pytest.approx(qty)
    assert row["side"] == side
    assert row["filled_qty"] == pytest.approx(qty)
    assert row["status"] == "filled"


@given(
    qty=st.integers(1, 10),
    extra=st.integers(1, 5),
    price=st.integers(10, 30),
    retries=st.integers(1, 4),
    client=_CLIENT,
)
@_FAST
def test_loopback_duplicate_client_id_fills_once(
    qty: int, extra: int, price: int, retries: int, client: str
) -> None:
    broker = LoopbackBroker(cash=1_000_000.0)
    first = LiveOrder(client, "AAPL", "buy", float(qty), float(price), 1)
    ack = broker.submit(first)
    assert ack.status == "filled"
    assert ack.filled_quantity == pytest.approx(qty)
    for attempt in range(retries):
        retry = LiveOrder(client, "AAPL", "sell", float(qty + extra), float(price + 1), 2 + attempt)
        again = broker.submit(retry)
        assert again is ack
    assert broker.positions["AAPL"] == pytest.approx(qty)
    assert broker.cash == pytest.approx(1_000_000.0 - qty * price)
    assert len(broker.fills(broker_to_client={}).fills) == 1
    assert broker.recover(client) is ack


@given(
    qty=st.integers(1, 8),
    price=st.integers(10, 40),
    retries=st.integers(1, 4),
    client=_CLIENT,
)
@_FAST
def test_loopback_rejected_client_id_retries_stay_rejected(
    qty: int, price: int, retries: int, client: str
) -> None:
    broker = LoopbackBroker(cash=500.0, positions={"AAPL": 1.0})
    sell = LiveOrder(client, "AAPL", "sell", float(qty + 1), float(price), 1)
    ack = broker.submit(sell)
    assert ack.status == "rejected"
    assert ack.reason == "insufficient position"
    for attempt in range(retries):
        retry = LiveOrder(client, "AAPL", "buy", float(qty), float(price), 2 + attempt)
        assert broker.submit(retry) is ack
    assert broker.positions["AAPL"] == pytest.approx(1.0)
    assert broker.cash == pytest.approx(500.0)
    assert broker.fills(broker_to_client={}).fills == ()


@given(
    qty=st.integers(1, 10),
    extra=st.integers(1, 4),
    price=st.integers(10, 30),
    retries=st.integers(1, 3),
)
@_FAST
def test_paper_adapter_duplicate_order_id_does_not_double(
    qty: int, extra: int, price: int, retries: int
) -> None:
    with TemporaryDirectory() as raw:
        adapter = PaperBrokerAdapter(
            state_path=Path(raw) / "paper.json",
            endpoint_kind="paper",
            starting_cash=_CASH,
        )
        assert adapter.submit(_order("o1", "BUY", qty)) == "accepted"
        for _ in range(retries):
            assert adapter.submit(_order("o1", "SELL", qty + extra)) == "duplicate_order"
        assert adapter.apply_fill(_fill("f1", qty, price, seconds=1)) == "accepted"
        assert adapter.apply_fill(_fill("f1", qty, price, seconds=1)) == "duplicate_fill"
        report = adapter.reconcile()
        assert report["n_orders"] == 1
        assert report["counters"]["duplicate_orders"] == retries
        assert report["computed_positions"]["A"] == pytest.approx(qty)
        assert adapter.position("A") == pytest.approx(qty)
        assert report["live_pnl_claim"] is False
        assert report["research_only"] is True


@given(
    qty=st.integers(1, 8),
    extra=st.integers(1, 4),
    price=st.integers(10, 25),
    retries=st.integers(1, 3),
    client=_CLIENT,
)
@_FAST
def test_desk_duplicate_client_id_returns_the_first_fill(
    qty: int, extra: int, price: int, retries: int, client: str
) -> None:
    with TemporaryDirectory() as raw:
        desk = _open_desk(Path(raw))
        try:
            desk.arm("loopback", "", TS_NS)
            first = desk.submit(
                "aapl",
                "BUY",
                float(qty),
                float(price),
                TS_NS + 1,
                client_order_id=client,
            )
            order = first["order"]
            assert isinstance(order, dict)
            assert first["status"] == "filled"
            assert "idempotent" not in first
            assert order["filled_qty"] == pytest.approx(qty)
            assert first["live_pnl_claim"] is False
            assert first["data_label"] == "SYNTHETIC"
            for attempt in range(retries):
                again = desk.submit(
                    "aapl",
                    "SELL",
                    float(qty + extra),
                    float(price + 1),
                    TS_NS + 2 + attempt,
                    client_order_id=client,
                )
                replay = again["order"]
                assert isinstance(replay, dict)
                assert again["idempotent"] is True
                assert again["accepted"] is True
                assert replay["order_id"] == order["order_id"]
                assert replay["qty"] == pytest.approx(qty)
                assert replay["side"] == "BUY"
                assert replay["filled_qty"] == pytest.approx(qty)
                assert again["positions"] == {"AAPL": float(qty)}
            assert desk.cash == pytest.approx(1_000_000.0 - qty * price)
            assert len(desk.orders) == 1
        finally:
            desk.close()


def _sim_order(
    order_id: str,
    side: OrderSide,
    qty: int,
    *,
    limit: float | None = None,
) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        security_id="A",
        symbol="A",
        side=side,
        quantity=float(qty),
        signal_time=_T0,
        decision_time=_T0,
        order_time=_T0,
        status=OrderStatus.NEW,
        limit_price=limit,
    )


@given(
    qty=st.integers(1, 12),
    extra=st.integers(1, 6),
    price=st.integers(10, 40),
    retries=st.integers(1, 4),
    side=st.sampled_from((OrderSide.BUY, OrderSide.SELL)),
    order_id=_CLIENT,
    rest=st.booleans(),
)
@_FAST
def test_simulated_broker_duplicate_order_id_applies_once(
    qty: int,
    extra: int,
    price: int,
    retries: int,
    side: OrderSide,
    order_id: str,
    rest: bool,
) -> None:
    """``order_id`` is the simulated broker's client-order key.

    A market fill and a resting limit both keep the first record across
    retries and across ``from_state``. The retry's side and size are ignored.
    """
    cfg = research_config()
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    broker = SimulatedBroker(config=cfg, initial_cash=_CASH)
    broker.mark({"A": float(price)})
    limit = float(price) if rest else None
    first = broker.submit(
        _sim_order(order_id, side, qty, limit=limit),
        price=float(price),
        nav=_CASH,
        adv_dollars=1e12,
        sigma=0.0,
    )
    if rest:
        assert first.fill is None
        assert first.order.status is OrderStatus.ACKED
    else:
        assert first.fill is not None
        assert first.order.status is OrderStatus.FILLED
    cash = broker.cash
    shares = dict(broker.shares)
    n_fills = len(broker.fills)
    n_history = len(broker.history)
    other = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY
    for _ in range(retries):
        again = broker.submit(
            _sim_order(order_id, other, qty + extra, limit=float(price + 1)),
            price=float(price + 1),
            nav=_CASH,
            adv_dollars=1e12,
            sigma=0.0,
        )
        assert again is first
        assert again.order.side is side
        assert again.order.quantity == pytest.approx(qty if rest else first.order.quantity)
    assert broker.cash == pytest.approx(cash)
    assert broker.shares == shares
    assert len(broker.fills) == n_fills
    assert len(broker.history) == n_history
    if not rest:
        signed = float(qty if side is OrderSide.BUY else -qty)
        assert broker.shares.get("A", 0.0) == pytest.approx(signed)

    restored = SimulatedBroker.from_state(cfg, broker.to_dict())
    replay = restored.submit(
        _sim_order(order_id, other, qty + extra),
        price=float(price),
        nav=_CASH,
        adv_dollars=1e12,
        sigma=0.0,
    )
    assert replay.order.order_id == order_id
    assert replay.order.side is side
    assert len(restored.fills) == n_fills
    assert len(restored.history) == n_history
    assert restored.shares == shares
    assert restored.cash == pytest.approx(cash)
    if rest:
        assert replay.fill is None
        assert order_id in restored.open_orders
    else:
        assert replay.fill is not None
        assert replay.fill.fill_id == first.fill.fill_id

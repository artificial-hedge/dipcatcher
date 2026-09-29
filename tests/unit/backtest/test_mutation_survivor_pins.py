"""Mutation-audit survivor pins — money path.

Hand-applied mutants at these lines passed the module's existing test slice
green; each test below encodes the intended contract and kills the mutant.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.event_sim.fills import advance_queue, bar_vwap_price, walk_book
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.loader import load_config
from quant_fund.config.models import (
    AppConfig,
    CostConfig,
    ExecutionConfig,
    FillConvention,
    KillSwitchConfig,
    RiskGateConfig,
)
from quant_fund.execution import costs, impact
from quant_fund.execution.almgren_chriss import almgren_chriss_trajectory
from quant_fund.execution.implementation_shortfall import aggregate_shortfall, shortfall_frame
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _fr_bars(n_days: int, *, vol_20: float | None = 0.02) -> pl.DataFrame:
    px = 100.0
    return pl.DataFrame(
        {
            "security_id": ["A"] * n_days,
            "event_time": [T0 + timedelta(days=i) for i in range(n_days)],
            "open": [px] * n_days,
            "close": [px] * n_days,
            "close_total_return": [px] * n_days,
            "volume": [1_000_000.0] * n_days,
            "adv": [px * 1_000_000.0] * n_days,
            "vol_20": [vol_20] * n_days,
            "source": ["file"] * n_days,
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def _fr_weights(n_days: int, w: float = 0.5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + timedelta(days=i) for i in range(n_days)],
            "security_id": ["A"] * n_days,
            "target_weight": [w] * n_days,
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def _fr_cfg(**costs_kw: float) -> AppConfig:
    defaults: dict[str, float | bool] = {
        "commission_bps": 0.0,
        "half_spread_bps": 0.0,
        "impact_y": 0.0,
        "bps_per_turnover": 0.0,
        "borrow_bps_per_year": 0.0,
        "frictionless": False,
        "participation_limit": 1.0,
    }
    defaults.update(costs_kw)
    return AppConfig(
        costs=CostConfig(**defaults),  # type: ignore[arg-type]
        risk_gate=RiskGateConfig(
            max_order_notional=1e12,
            max_gross=100.0,
            max_net=100.0,
            max_name=1.0,
            max_participation=1.0,
            max_predicted_vol=100.0,
            stale_price_bars=3,
            stale_model_hours=1e9,
        ),
        execution=ExecutionConfig(fill=FillConvention.NEXT_OPEN),
        kill_switch=KillSwitchConfig(state="ENABLED"),
    )


def test_nav_zero_break_halts_replay() -> None:
    """nav <= 0 halts the loop — a dead book must not keep emitting NAV rows."""
    res = run_backtest_fast(_fr_bars(5), _fr_weights(5), _fr_cfg(), initial_nav=0.0)
    assert res.equity.height == 0


def test_missing_vol_still_charges_impact() -> None:
    """A null vol_20 falls back to a conservative sigma, never to zero impact."""
    bars = _fr_bars(6, vol_20=float("nan"))
    res = run_backtest_fast(bars, _fr_weights(6), _fr_cfg(impact_y=0.5))
    impact_val = res.metrics["impact"]
    assert isinstance(impact_val, (int, float))
    assert impact_val > 0.0


def test_book_walk_vwap_uses_filled_not_displayed() -> None:
    """VWAP divides notional by shares actually filled, not displayed size."""
    res = walk_book("buy", 5.0, [(100.0, 10.0), (101.0, 10.0)])
    assert res.filled == 5.0
    assert res.vwap == pytest.approx(100.0)


def test_queue_fill_capped_by_prints_that_cleared_queue() -> None:
    """Tape volume bounds both queue consumption and our slice of the touch."""
    step = advance_queue(
        side="buy",
        remaining=1e6,
        limit_price=100.0,
        queue_ahead=50.0,
        touch_price=100.0,
        touch_size=100.0,
        traded_volume=60.0,
    )
    # 60 prints consume the 50 ahead of us; only 10 reach our order.
    assert step.filled == pytest.approx(10.0)


def test_inverted_bar_falls_through_to_close() -> None:
    """A corrupt high < low bar must not fabricate a typical price."""
    row: dict[str, object] = {"high": 99.0, "low": 101.0, "close": 105.0}
    assert bar_vwap_price(row) == 105.0


def _broker_cfg(tmp_path) -> AppConfig:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(
    side: OrderSide = OrderSide.BUY,
    qty: float = 10.0,
    oid: str = "o1",
    limit: float | None = None,
) -> Order:
    t = datetime(2024, 1, 2, tzinfo=UTC)
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        limit_price=limit,
        signal_time=t,
        decision_time=t,
        order_time=t,
        status=OrderStatus.NEW,
    )


def test_buy_limit_touched_at_bar_low_fills(tmp_path) -> None:
    """lo <= limit: a touch at the limit is executable, not one tick past it."""
    broker = SimulatedBroker(config=_broker_cfg(tmp_path), initial_cash=1e6)
    rec = broker.submit(
        _order(qty=1.0, limit=100.0),
        price=100.0,
        nav=1e6,
        adv_dollars=1e9,
        bar_open=105.0,
        bar_high=110.0,
        bar_low=100.0,
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.price == 100.0


def test_buy_limit_gap_open_gets_price_improvement(tmp_path) -> None:
    """Buy limit fills at min(open, limit) — the open's better price wins."""
    broker = SimulatedBroker(config=_broker_cfg(tmp_path), initial_cash=1e6)
    rec = broker.submit(
        _order(qty=1.0, limit=100.0),
        price=100.0,
        nav=1e6,
        adv_dollars=1e9,
        bar_open=90.0,
        bar_high=110.0,
        bar_low=85.0,
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.price == 90.0


def test_sell_limit_touched_at_bar_high_fills(tmp_path) -> None:
    """h >= limit: symmetric touch-fill for sells at the bar high."""
    broker = SimulatedBroker(config=_broker_cfg(tmp_path), initial_cash=1e6)
    rec = broker.submit(
        _order(side=OrderSide.SELL, qty=1.0, limit=100.0),
        price=100.0,
        nav=1e6,
        adv_dollars=1e9,
        bar_open=95.0,
        bar_high=100.0,
        bar_low=90.0,
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.price == 100.0


def test_cash_check_includes_fees_not_just_notional(tmp_path) -> None:
    """A buy needs cash >= notional + costs; fee-blind checks overspend cash."""
    cfg = _broker_cfg(tmp_path)
    cfg.costs.commission_bps = 10.0
    broker = SimulatedBroker(config=cfg, initial_cash=1000.5)
    rec = broker.submit(_order(qty=10.0), price=100.0, nav=1000.5, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.INSUFFICIENT_CASH.value


def test_favorable_drift_records_zero_slippage(tmp_path) -> None:
    """Slippage is adverse-only: fills better than decision_price report 0."""
    broker = SimulatedBroker(config=_broker_cfg(tmp_path), initial_cash=1e6)
    rec = broker.submit(
        _order(qty=1.0), price=100.0, nav=1e6, adv_dollars=1e9, decision_price=105.0
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.slippage == 0.0


def test_partial_limit_fill_rests_true_residual(tmp_path) -> None:
    """The residual is requested - executed, not requested + executed."""
    cfg = _broker_cfg(tmp_path)
    cfg.costs.participation_limit = 0.1
    broker = SimulatedBroker(config=cfg, initial_cash=1e6)
    rec = broker.submit(
        _order(qty=2000.0, limit=200.0),
        price=100.0,
        nav=1e6,
        adv_dollars=1e6,
        bar_open=100.0,
        bar_high=110.0,
        bar_low=90.0,
    )
    assert rec.order.status is OrderStatus.FILLED
    residual = broker.open_orders["o1"]
    assert residual.status is OrderStatus.PARTIAL
    assert residual.quantity == pytest.approx(1000.0)


def test_sqrt_impact_is_sqrt_not_linear() -> None:
    """Impact scales with sqrt(participation): participation=1/400 -> sqrt=1/20."""
    got = costs.sqrt_impact(
        quantity=1.0, price=100.0, adv_dollars=40_000.0, sigma=0.02, upsilon=0.5
    )
    # notional=100, participation=0.0025, sqrt=0.05 -> 100*0.5*0.02*0.05
    assert got == pytest.approx(0.05)


def test_required_participation_slices_to_max_bars() -> None:
    """max_bars truncates the forecast window; oversized orders fail closed."""
    with pytest.raises(ValueError, match="exceeds forecast volume"):
        impact.required_participation(
            quantity=150.0, volume_forecast=np.array([100.0, 100.0, 100.0]), max_bars=1
        )


def test_aggregate_shortfall_favorable_is_negative_side_only() -> None:
    """favorable_is sums total_is < 0 fills, not the unfavorable side."""
    fills = pl.DataFrame(
        {
            "security_id": ["A", "B"],
            "quantity": [1.0, 1.0],
            "decision_price": [100.0, 100.0],
            "price": [99.0, 101.0],
        }
    )
    agg = aggregate_shortfall(shortfall_frame(fills))
    assert agg["favorable_is"] < 0.0


def test_ac_trajectory_larger_eta_trades_slower() -> None:
    """kappa^2 = lambda*sigma^2/eta: bigger temp impact defers trading."""
    slow = almgren_chriss_trajectory(1.0, 4, sigma=0.02, eta=4e-6, gamma=1e-7, risk_aversion=1.0)
    fast = almgren_chriss_trajectory(1.0, 4, sigma=0.02, eta=1e-6, gamma=1e-7, risk_aversion=1.0)
    assert slow[1] > fast[1]

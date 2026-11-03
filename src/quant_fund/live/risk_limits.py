"""Pure pre-trade limit decisions for the live gateway.

These checks run before a broker submit. They do not place orders and they
do not claim a profit result. A mutation score on this module is a test
diagnostic, not market evidence.
"""

from __future__ import annotations

import math

from quant_fund.pretrade.config import LimitConfig
from quant_fund.pretrade.engine import PretradeEngine

# Smaller value is a tighter ceiling.
_SMALLER_IS_TIGHTER: tuple[str, ...] = (
    "max_order_notional",
    "max_order_quantity",
    "price_collar_bps",
    "max_position_notional",
    "max_position_quantity",
    "max_gross_notional",
    "max_net_notional",
    "max_name_concentration",
    "max_daily_loss_fraction",
    "max_trailing_drawdown_fraction",
    "max_orders_per_window",
    "max_messages_per_window",
    "max_reference_age_ns",
    "max_mark_age_ns",
    "pdt_max_day_trades",
)

# A coarser tick rejects more prices, so a larger tick is tighter.
_LARGER_IS_TIGHTER: tuple[str, ...] = ("qty_tick", "price_tick")

# Changing these reallocates the hot book. Refuse them while the process is up.
_STRUCTURAL: tuple[str, ...] = (
    "rate_window_ns",
    "duplicate_window_ns",
    "pdt_equity_threshold",
    "pdt_window_sessions",
    "settlement_ns",
    "max_lots_per_symbol",
    "ring_capacity",
    "symbol_capacity",
    "day_trade_capacity",
    "unsettled_capacity",
)


def on_tick(value: float, tick: float) -> bool:
    """True when ``value`` is a positive multiple of ``tick``."""
    if not math.isfinite(value) or not math.isfinite(tick) or tick <= 0.0 or value <= 0.0:
        return False
    steps = round(value / tick)
    return abs(value - (steps * tick)) <= tick * 1e-6


def precision_reasons(
    qty: float, price: float, qty_tick: float, price_tick: float
) -> tuple[str, ...]:
    """Names of price or quantity ticks the order misses. Empty means aligned."""
    reasons: list[str] = []
    if not on_tick(qty, qty_tick):
        reasons.append("qty_precision")
    if not on_tick(price, price_tick):
        reasons.append("price_precision")
    return tuple(reasons)


def loss_reason(nav: float, daily_floor: float, dd_floor: float) -> str:
    """``daily_loss`` or ``trailing_drawdown`` when nav is on or through a floor.

    A non-finite floor does not trip. Daily loss wins when both floors bind.
    """
    if math.isfinite(daily_floor) and nav <= daily_floor:
        return "daily_loss"
    if math.isfinite(dd_floor) and nav <= dd_floor:
        return "trailing_drawdown"
    return ""


def loosened_names(current: LimitConfig, proposed: LimitConfig) -> tuple[str, ...]:
    """Ceilings that got looser. Equal values are not looser."""
    names: list[str] = []
    _collect_smaller(names, current, proposed)
    _collect_larger(names, current, proposed)
    _collect_flags(names, current, proposed)
    return tuple(names)


def structural_changes(current: LimitConfig, proposed: LimitConfig) -> tuple[str, ...]:
    """Fields that cannot be swapped on a live book."""
    changed = [name for name in _STRUCTURAL if getattr(current, name) != getattr(proposed, name)]
    return tuple(changed)


def install_ceilings(engine: PretradeEngine, limits: LimitConfig) -> None:
    """Copy ceilings onto the running book. Rings and lots stay put."""
    book = engine.book
    book.max_qty = float(limits.max_order_quantity)
    book.max_notional = float(limits.max_order_notional)
    book.max_pos_qty = float(limits.max_position_quantity)
    book.max_pos_notional = float(limits.max_position_notional)
    book.max_gross = float(limits.max_gross_notional)
    book.max_net = float(limits.max_net_notional)
    book.max_orders = int(limits.max_orders_per_window)
    book.max_msgs = int(limits.max_messages_per_window)
    book.cash_account = 1 if limits.cash_account else 0
    book.restrict_settled = 1 if limits.restrict_to_settled_cash else 0
    book.qty_tick = float(limits.qty_tick)
    book.price_tick = float(limits.price_tick)
    book.collar_frac = float(limits.price_collar_bps) / 10_000.0
    book.pdt_max = int(limits.pdt_max_day_trades)
    engine._daily_loss = float(limits.max_daily_loss_fraction)
    engine._trailing_dd = float(limits.max_trailing_drawdown_fraction)
    engine._concentration = float(limits.max_name_concentration)
    engine._max_ref_age = int(limits.max_reference_age_ns)
    engine._max_mark_age = int(limits.max_mark_age_ns)
    engine._recompute_floors()
    nav = float(book.nav)
    if nav == nav and nav > 0.0:
        book.name_limit = float(limits.max_name_concentration) * nav
    engine.config = engine.config.model_copy(update={"limits": limits})


def _collect_smaller(names: list[str], current: LimitConfig, proposed: LimitConfig) -> None:
    for name in _SMALLER_IS_TIGHTER:
        if float(getattr(proposed, name)) > float(getattr(current, name)):
            names.append(name)


def _collect_larger(names: list[str], current: LimitConfig, proposed: LimitConfig) -> None:
    for name in _LARGER_IS_TIGHTER:
        if float(getattr(proposed, name)) < float(getattr(current, name)):
            names.append(name)


def _collect_flags(names: list[str], current: LimitConfig, proposed: LimitConfig) -> None:
    if current.cash_account and not proposed.cash_account:
        names.append("cash_account")
    if current.restrict_to_settled_cash and not proposed.restrict_to_settled_cash:
        names.append("restrict_to_settled_cash")

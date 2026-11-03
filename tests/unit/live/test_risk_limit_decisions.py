"""Decision tests for live risk ceilings. A mutation score is not market evidence."""

from __future__ import annotations

from quant_fund.live.risk_limits import (
    install_ceilings,
    loosened_names,
    loss_reason,
    on_tick,
    precision_reasons,
    structural_changes,
)
from tests.unit.pretrade.support import limit_config, make_engine


def test_on_tick_accepts_exact_multiples_and_rejects_the_neighbors() -> None:
    assert on_tick(10.0, 0.0001) is True
    assert on_tick(99.5, 0.0001) is True
    assert on_tick(10.00003, 0.0001) is False
    assert on_tick(100.00003, 0.0001) is False
    assert on_tick(0.0, 0.0001) is False
    assert on_tick(1.0, 0.0) is False
    assert on_tick(float("nan"), 0.0001) is False


def test_precision_reasons_name_each_off_tick_field() -> None:
    assert precision_reasons(10.0, 100.0, 0.0001, 0.0001) == ()
    assert precision_reasons(10.00003, 100.0, 0.0001, 0.0001) == ("qty_precision",)
    assert precision_reasons(10.0, 100.00003, 0.0001, 0.0001) == ("price_precision",)
    assert precision_reasons(1.5, 2.5, 1.0, 1.0) == ("qty_precision", "price_precision")


def test_loss_reason_trips_on_the_floor_and_ignores_a_non_finite_floor() -> None:
    assert loss_reason(980_000.0, 980_000.0, 950_000.0) == "daily_loss"
    assert loss_reason(980_000.01, 980_000.0, 950_000.0) == ""
    assert loss_reason(990_000.0, 980_000.0, 997_500.0) == "trailing_drawdown"
    assert loss_reason(990_000.0, 997_500.0, 997_500.0) == "daily_loss"
    assert loss_reason(10.0, float("inf"), float("inf")) == ""
    assert loss_reason(10.0, float("nan"), 15.0) == "trailing_drawdown"


def test_equal_ceilings_are_not_looser_and_a_raised_cap_is() -> None:
    current = limit_config()
    assert loosened_names(current, current) == ()
    tighter = limit_config(max_order_quantity=5)
    assert loosened_names(current, tighter) == ()
    looser = limit_config(max_order_quantity=current.max_order_quantity + 1)
    assert loosened_names(current, looser) == ("max_order_quantity",)
    coarse = limit_config(qty_tick=current.qty_tick * 10)
    assert loosened_names(current, coarse) == ()
    finer = limit_config(qty_tick=current.qty_tick / 10)
    assert loosened_names(current, finer) == ("qty_tick",)


def test_dropping_cash_or_settled_cash_rules_is_a_loosen() -> None:
    current = limit_config(cash_account=True, restrict_to_settled_cash=True)
    assert "cash_account" in loosened_names(current, limit_config(cash_account=False))
    assert loosened_names(current, limit_config(cash_account=True)) == ()
    margin = limit_config(cash_account=False, restrict_to_settled_cash=False)
    assert loosened_names(margin, limit_config(cash_account=True)) == ()
    assert "restrict_to_settled_cash" in loosened_names(
        current, limit_config(restrict_to_settled_cash=False)
    )
    assert loosened_names(margin, margin) == ()


def test_structural_fields_are_named_and_equal_configs_are_empty() -> None:
    current = limit_config()
    assert structural_changes(current, current) == ()
    changed = limit_config(ring_capacity=128)
    assert structural_changes(current, changed) == ("ring_capacity",)


def test_install_ceilings_updates_the_running_book_and_the_daily_floor() -> None:
    engine = make_engine(max_daily_loss_fraction=0.02, max_order_quantity=100)
    install_ceilings(engine, limit_config(max_order_quantity=5, max_daily_loss_fraction=0.10))
    assert engine.book.max_qty == 5
    assert engine.book.daily_floor == engine.book.session_start * 0.90
    assert engine.config.limits.max_order_quantity == 5

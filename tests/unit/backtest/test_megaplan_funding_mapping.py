"""SYNTHETIC compatibility pins for explicit coarse-bar funding mappings."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from scripts.megaplan_eval import Evaluator

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.perp_engine import _funding_by_time, run_perp_backtest
from quant_fund.config.loader import load_config

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _inputs():
    times = [T0 + timedelta(days=i) for i in range(3)]
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 3,
            "event_time": times,
            "open": [100.0] * 3,
            "close": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "volume": [1e6] * 3,
            "source": ["synthetic"] * 3,
        }
    )
    funding = pl.DataFrame(
        {
            "security_id": ["A"] * 3,
            "event_time": [times[1] + timedelta(hours=h) for h in (0, 8, 16)],
            "value": [0.0001, -0.0002, 0.0003],
        }
    )
    weights = pl.DataFrame({"security_id": ["A"], "event_time": [T0], "target_weight": [0.5]})
    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_order_notional = 1e12
    return bars, funding, weights, cfg


def test_daily_evaluator_preserves_settlements_and_signed_cashflows():
    bars, funding, weights, cfg = _inputs()
    evaluator = Evaluator(bars, funding, cfg, 100_000.0, 2, 365.25)
    mapped = evaluator._funding_on_bars(funding)
    assert mapped["event_time"].to_list() == funding["event_time"].to_list()
    assert mapped["application_time"].to_list() == [T0 + timedelta(days=1)] * 3
    result = run_perp_backtest(bars, mapped, weights, cfg, initial_nav=100_000.0)
    assert result.metrics["funding_paid_total"] == pytest.approx(20.0)
    assert result.metrics["funding_received_total"] == pytest.approx(10.0)
    assert result.metrics["funding_net"] == pytest.approx(-10.0)
    assert result.metrics["funding_events_applied"] == 3
    assert result.metrics["funding_events_dropped"] == 0
    assert result.equity["nav"][-1] == pytest.approx(99_990.0)
    metrics = evaluator.run(weights, T0, T0 + timedelta(days=3), None)
    assert metrics["funding_paid_total"] == pytest.approx(20.0)
    assert metrics["funding_received_total"] == pytest.approx(10.0)
    assert metrics["funding_events_dropped"] == 0


def test_mapping_does_not_hide_duplicate_source_settlements():
    bars, funding, weights, cfg = _inputs()
    funding = pl.concat([funding, funding.head(1)])
    evaluator = Evaluator(bars, funding, cfg, 100_000.0, 2, 365.25)
    with pytest.raises(ValueError, match="duplicate funding event"):
        evaluator.run(weights, T0, T0 + timedelta(days=3), None)


def test_raw_funding_retains_exact_timestamp_contract():
    bars, funding, weights, cfg = _inputs()
    result = run_perp_backtest(bars, funding, weights, cfg, initial_nav=100_000.0)
    assert result.metrics["funding_events_applied"] == 1
    assert result.metrics["funding_events_dropped"] == 2
    assert result.metrics["funding_paid_total"] == pytest.approx(5.0)
    assert result.metrics["funding_received_total"] == 0.0


@pytest.mark.parametrize("mapped_time", [None, "2024-01-01", T0 + timedelta(days=2)])
def test_invalid_or_future_application_time_fails_closed(mapped_time):
    _, funding, _, _ = _inputs()
    mapped = funding.head(1).with_columns(pl.lit(mapped_time).alias("application_time"))
    with pytest.raises(ValueError, match="application_time"):
        _funding_by_time(mapped)


def test_unmatched_explicit_application_label_fails_closed():
    bars, funding, weights, cfg = _inputs()
    mapped = funding.with_columns(pl.lit(T0 + timedelta(hours=12)).alias("application_time"))
    with pytest.raises(ValueError, match="input bar label"):
        run_perp_backtest(bars, mapped, weights, cfg)


def test_duplicate_source_rejected_even_when_mapped_to_different_bars():
    _, funding, _, _ = _inputs()
    mapped = pl.concat([funding.head(1), funding.head(1)]).with_columns(
        pl.Series("application_time", [T0, T0 + timedelta(days=1)])
    )
    with pytest.raises(ValueError, match="duplicate funding event"):
        _funding_by_time(mapped)


def test_carry_mapping_keeps_gross_cashflows_and_rejects_missing_label():
    bars, funding, weights, cfg = _inputs()
    evaluator = Evaluator(bars, funding, cfg, 100_000.0, 2, 365.25)
    mapped = evaluator._funding_on_bars(funding)
    result = run_carry_backtest(bars, bars, mapped, weights, cfg, initial_nav=100_000.0)
    assert result.metrics["funding_paid_total"] == pytest.approx(10.0)
    assert result.metrics["funding_received_total"] == pytest.approx(20.0)
    assert result.metrics["funding_events_applied"] == 3
    assert result.metrics["funding_events_dropped"] == 0
    mapped = mapped.with_columns(pl.lit(T0 + timedelta(hours=12)).alias("application_time"))
    with pytest.raises(ValueError, match="input bar label"):
        run_carry_backtest(bars, bars, mapped, weights, cfg)


def test_mapping_cannot_move_settlement_to_arbitrary_older_bar():
    bars, funding, weights, cfg = _inputs()
    mapped = funding.with_columns(pl.lit(T0).alias("application_time"))
    with pytest.raises(ValueError, match="latest bar"):
        run_perp_backtest(bars, mapped, weights, cfg)


def test_exact_grid_and_empty_mapping_preserve_inputs():
    bars, funding, _, cfg = _inputs()
    evaluator = Evaluator(bars, funding, cfg, 100_000.0, 2, 365.25)
    assert evaluator._funding_on_bars(funding.clear()).equals(funding.clear())
    exact = funding.head(1)
    mapped = evaluator._funding_on_bars(exact)
    assert mapped["application_time"].to_list() == exact["event_time"].to_list()


def test_application_timezone_must_be_comparable():
    _, funding, _, _ = _inputs()
    mapped = funding.with_columns(pl.lit(T0.replace(tzinfo=None)).alias("application_time"))
    with pytest.raises(ValueError, match="comparable"):
        _funding_by_time(mapped)

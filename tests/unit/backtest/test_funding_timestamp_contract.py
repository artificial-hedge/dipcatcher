"""SYNTHETIC contract checks; no evidence of market performance."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.perp_engine import run_perp_backtest
from quant_fund.config.loader import load_config

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A"] * 6,
            "event_time": [T0 + timedelta(hours=i) for i in range(6)],
            "open": [100.0] * 6,
            "high": [110.0] * 6,
            "low": [100.0] * 6,
            "close": [110.0] * 6,
            "volume": [1e6] * 6,
            "source": ["synthetic"] * 6,
        }
    )


def _funding(hours: list[float], sid: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [sid] * len(hours),
            "event_time": [T0 + timedelta(hours=h) for h in hours],
            "value": [0.01] * len(hours),
            # Settlement valuation uses the bar close, not this optional field.
            "mark_price": [999.0] * len(hours),
        }
    )


@pytest.fixture(params=["perp", "carry"])
def run_book(request, tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_order_notional = 1e12

    def run(funding, *, exit_at=None, enabled=True, bars=None):
        cfg.perp.funding_enabled = enabled
        times, targets = [T0], [0.5]
        if exit_at is not None:
            times.append(T0 + timedelta(hours=exit_at))
            targets.append(0.0)
        weights = pl.DataFrame(
            {"event_time": times, "security_id": ["A"] * len(times), "target_weight": targets}
        )
        bars = _bars() if bars is None else bars
        if request.param == "perp":
            return run_perp_backtest(bars, funding, weights, cfg, initial_nav=1e5)
        return run_carry_backtest(bars, bars, funding, weights, cfg, initial_nav=1e5)

    return run


def test_only_exact_labels_settle_and_off_grid_rows_are_counted(run_book):
    # At 01:00 the open fill establishes 500 units, valued for funding at 110.
    baseline = run_book(_funding([1]))
    # Before/after the panel and either side of an exact label are all off-grid.
    mixed = run_book(_funding([-1, 0.999, 1, 1.001, 6]))
    assert baseline.metrics["funding_events_applied"] == 1
    assert baseline.metrics["funding_events_dropped"] == 0
    assert mixed.metrics["funding_events_applied"] == 1
    assert mixed.metrics["funding_events_dropped"] == 4
    assert mixed.equity.equals(baseline.equity)
    assert abs(mixed.metrics["funding_net"]) == pytest.approx(550.0)


def test_funding_on_entry_and_exit_uses_post_fill_position(run_book):
    # Signal at 02:00 exits at 03:00. Funding at 02:00 still applies;
    # funding at 03:00 sees a flat book, as does 00:00 before entry.
    result = run_book(_funding([0, 1, 2, 3]), exit_at=2)
    assert result.metrics["funding_events_applied"] == 2
    assert result.metrics["funding_events_dropped"] == 0
    assert abs(result.metrics["funding_net"]) == pytest.approx(1100.0)


def test_dropped_count_checks_global_timestamp_not_symbol(run_book):
    # B has no bar or position, but its timestamp appears for A.
    result = run_book(_funding([1], sid="B"))
    assert result.metrics["funding_events_applied"] == 0
    assert result.metrics["funding_events_dropped"] == 0
    assert result.metrics["funding_net"] == 0.0


def test_disabled_funding_does_not_count_off_grid_rows(run_book):
    result = run_book(_funding([1, 1.5]), enabled=False)
    assert result.metrics["funding_events_applied"] == 0
    assert result.metrics["funding_events_dropped"] == 0
    assert result.metrics["funding_net"] == 0.0


def test_funding_uses_permitted_carried_close_for_missing_symbol_bar(run_book):
    bars = _bars()
    # A has no 02:00 bar, but B keeps that timestamp on the execution grid.
    sparse = pl.concat(
        [
            bars.filter(pl.col("event_time") != T0 + timedelta(hours=2)),
            bars.with_columns(pl.lit("B").alias("security_id")),
        ]
    )
    result = run_book(_funding([2]), bars=sparse)
    assert result.metrics["funding_events_applied"] == 1
    assert result.metrics["funding_events_dropped"] == 0
    assert abs(result.metrics["funding_net"]) == pytest.approx(550.0)

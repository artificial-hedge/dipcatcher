"""Published listing ends must be flat before the last print ages out."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.backtest.engine import StaleValuationError, run_backtest
from quant_fund.backtest.listing_exit import (
    ListingExitError,
    apply_published_listing_exits,
    schedule_listing_exits,
)
from quant_fund.cli.main import app
from quant_fund.config.loader import load_config

pytestmark = pytest.mark.e2e_smoke

runner = CliRunner()


def _dates() -> list[datetime]:
    return [datetime(2024, 1, day, tzinfo=UTC) for day in range(1, 7)]


def _bars(dates: list[datetime]) -> pl.DataFrame:
    a_times = dates[:3]
    event_time = a_times + list(dates)
    n = len(event_time)
    return pl.DataFrame(
        {
            "security_id": ["A"] * len(a_times) + ["B"] * len(dates),
            "event_time": event_time,
            "open": [100.0] * n,
            "close": [100.0] * n,
            "close_total_return": [100.0] * n,
            "volume": [1_000_000.0] * n,
            "adv": [100_000_000.0] * n,
            "vol_20": [0.02] * n,
            "source": ["synthetic"] * n,
        }
    )


def _weights(dates: list[datetime], a_weight: float = 0.4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [dates[0], dates[0], dates[1], dates[1]],
            "security_id": ["A", "B", "A", "B"],
            "target_weight": [a_weight, 0.2, a_weight, 0.2],
        }
    )


def _master(valid_to: datetime, available: datetime, security_id: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [security_id],
            "valid_to": [valid_to],
            "available_time": [available],
        }
    )


def _target(frame: pl.DataFrame, security_id: str, when: datetime) -> float:
    rows = frame.filter((pl.col("security_id") == security_id) & (pl.col("event_time") == when))
    return float(rows.get_column("target_weight").item())


def test_schedule_flattens_held_name_on_the_decision_before_last_print() -> None:
    dates = _dates()
    out = schedule_listing_exits(_weights(dates), _bars(dates), _master(dates[3], dates[0]))
    assert _target(out, "A", dates[0]) == 0.4
    assert _target(out, "A", dates[1]) == 0.0
    assert _target(out, "B", dates[0]) == 0.2
    assert _target(out, "B", dates[1]) == 0.2


def test_schedule_lets_the_event_loop_exit_on_the_last_print() -> None:
    dates = _dates()
    bars = _bars(dates)
    weights = _weights(dates)
    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = True
    cfg.risk_gate.stale_price_bars = 1
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    with pytest.raises(StaleValuationError):
        run_backtest(bars, weights, cfg, fast=False)
    adjusted = schedule_listing_exits(weights, bars, _master(dates[3], dates[0]))
    result = run_backtest(bars, adjusted, cfg, fast=False)
    assert result.equity.height > 2
    assert result.fills.height >= 1
    nav = result.equity.get_column("nav").to_numpy()
    assert nav.min() > 0.0


def test_schedule_appends_a_flat_row_when_the_exit_decision_is_missing() -> None:
    dates = _dates()
    weights = pl.DataFrame(
        {
            "event_time": [dates[0], dates[0], dates[1]],
            "security_id": ["A", "B", "B"],
            "target_weight": [0.4, 0.2, 0.2],
            "alpha": [0.1, 0.2, 0.2],
        }
    )
    out = schedule_listing_exits(weights, _bars(dates), _master(dates[3], dates[0]))
    held = out.filter(pl.col("security_id") == "A").sort("event_time")
    assert held.get_column("event_time").to_list() == [dates[0], dates[1]]
    assert held.get_column("target_weight").to_list() == [0.4, 0.0]
    assert held.get_column("alpha").to_list()[1] is None


def test_schedule_leaves_an_already_flat_name() -> None:
    dates = _dates()
    weights = _weights(dates, a_weight=0.0)
    assert schedule_listing_exits(weights, _bars(dates), _master(dates[3], dates[0])) is weights


def test_schedule_leaves_an_end_outside_the_sample() -> None:
    dates = _dates()
    weights = _weights(dates)
    later = dates[-1] + timedelta(days=5)
    assert schedule_listing_exits(weights, _bars(dates), _master(later, dates[0])) is weights


def test_schedule_rejects_a_held_name_with_no_published_exit() -> None:
    dates = _dates()
    with pytest.raises(ListingExitError, match="published exit"):
        schedule_listing_exits(_weights(dates), _bars(dates), _master(dates[3], dates[2]))


def test_schedule_rejects_empty_weights_and_missing_columns() -> None:
    dates = _dates()
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime(time_zone="UTC"),
            "security_id": pl.Utf8,
            "target_weight": pl.Float64,
        }
    )
    with pytest.raises(ListingExitError, match="empty"):
        schedule_listing_exits(empty, _bars(dates), _master(dates[3], dates[0]))
    with pytest.raises(ListingExitError, match="missing columns"):
        schedule_listing_exits(
            _bars(dates).drop("close"), _bars(dates), _master(dates[3], dates[0])
        )


def test_apply_published_listing_exits_without_a_master_is_unchanged(tmp_path: Path) -> None:
    dates = _dates()
    weights = _weights(dates)
    assert apply_published_listing_exits(weights, _bars(dates), tmp_path) is weights


def test_backtest_command_flattens_a_published_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = _dates()
    feat = _bars(dates)
    weights = _weights(dates)
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    _master(dates[3], dates[0]).write_parquet(bronze / "security_master.parquet")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda _cfg: pl.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr("quant_fund.features.engine.build_features", lambda _bars, _cfg: feat)
    monkeypatch.setattr("quant_fund.pipeline.forecast.decision_dates", lambda _cfg, stamps: stamps)
    monkeypatch.setattr(
        "quant_fund.pipeline.forecast.build_causal_weight_panel",
        lambda _cfg, _dates: weights,
    )
    seen: dict[str, pl.DataFrame] = {}

    def _run(frame: pl.DataFrame, booked: pl.DataFrame, _cfg: object) -> SimpleNamespace:
        del frame
        seen["weights"] = booked
        return SimpleNamespace(source_note="SYNTHETIC", metrics={"total_return": 0.0})

    monkeypatch.setattr("quant_fund.backtest.engine.run_backtest", _run)
    result = runner.invoke(app, ["backtest", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    booked = seen["weights"]
    assert _target(booked, "A", dates[1]) == 0.0
    assert _target(booked, "B", dates[1]) == 0.2
    assert _target(booked, "A", dates[0]) == 0.4

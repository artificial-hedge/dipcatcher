"""Reconciliation mismatches and the paper clocks that bound a replay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.paper.clock import ReplayClock, WallClock
from quant_fund.paper.recon import reconcile_broker_states, reconcile_equity, reconcile_fills


def test_broker_reconciliation_flags_books_that_do_not_match() -> None:
    with pytest.raises(ValueError, match="dicts"):
        reconcile_broker_states([], {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        reconcile_broker_states({}, {}, tol=float("nan"))

    expected = {
        "cash": 10.0,
        "reject_count": 1,
        "shares": {"A": 2.0, "B": "nope"},
        "last_marks": {"A": 10.0, "C": 3.0},
        "kill_state": "ENABLED",
        "fill_convention": "next_open",
        "slot": "champion",
    }
    actual = {
        "cash": "missing",
        "reject_count": 4,
        "shares": {"A": 1.0, "B": 1.0},
        "last_marks": {"A": "bad", "D": 4.0},
        "kill_state": "HALT_NEW_ORDERS",
        "fill_convention": "next_open",
        "slot": "shadow",
    }
    report = reconcile_broker_states(expected, actual)
    assert report["match"] is False
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    text = " ".join(report["mismatches"])
    assert "cash" in text
    assert "reject_count" in text
    assert "shares[A]" in text
    assert "shares[B]" in text
    assert "last_marks[C]" in text
    assert "last_marks[A]" in text
    assert "kill_state" in text
    assert "slot" in text


def test_equity_and_fill_reconciliation_count_unmatched_rows() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        reconcile_equity(
            pl.DataFrame({"nav": [1.0]}), pl.DataFrame({"event_time": [1], "nav": [1.0]})
        )
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    t1 = datetime(2024, 1, 3, tzinfo=UTC)
    expected = pl.DataFrame({"event_time": [t0, t1], "nav": [1.0, None]})
    actual = pl.DataFrame({"event_time": [t1, t0], "nav": [None, 2.0]})
    shared = reconcile_equity(expected, actual)
    assert shared["match"] is False
    assert shared["matched_null_nav"] >= 1
    assert shared["live_pnl_claim"] is False

    clean_expected = pl.DataFrame({"event_time": [t0], "nav": [1.0]})
    clean_actual = pl.DataFrame({"event_time": [t0, t1], "nav": [1.5, 2.0]})
    drifted = reconcile_equity(clean_expected, clean_actual, tol=0.1)
    assert drifted["match"] is False
    assert drifted["unmatched_actual"] == 1
    assert drifted["max_abs_nav_delta"] == pytest.approx(0.5)
    assert drifted["research_only"] is True

    with pytest.raises(ValueError, match="fills missing"):
        reconcile_fills(pl.DataFrame({"security_id": ["A"]}), pl.DataFrame({"security_id": ["A"]}))
    fills_expected = pl.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "fill_time": [t0, t0, t1],
            "quantity": [1.0, 1.0, 2.0],
            "price": [10.0, 10.0, 11.0],
        }
    )
    fills_actual = pl.DataFrame(
        {
            "security_id": ["B", "C"],
            "fill_time": [t1, t0],
            "quantity": [3.0, 1.0],
            "price": [12.0, 9.0],
        }
    )
    fills = reconcile_fills(fills_expected, fills_actual)
    assert fills["match"] is False
    assert fills["n_qty_mismatch"] >= 1
    assert fills["n_price_mismatch"] >= 1
    assert fills["live_pnl_claim"] is False
    assert any("duplicated" in item for item in fills["mismatches"])
    assert any("unmatched" in item for item in fills["mismatches"])


def test_clocks_stop_at_the_end_of_the_replay() -> None:
    wall = WallClock()
    assert wall.exhausted() is False
    assert wall.tick() is not None
    stepped = WallClock(step=timedelta(seconds=1))
    first = stepped.tick()
    second = stepped.tick()
    assert first is not None and second is not None
    assert second - first == timedelta(seconds=1)

    empty = ReplayClock([])
    with pytest.raises(RuntimeError, match="no event times"):
        empty.now()
    assert empty.tick() is None
    assert empty.exhausted() is True
    assert empty.remaining() == 0
    assert list(empty) == []

    stamp = datetime(2024, 1, 2, tzinfo=UTC)
    replay = ReplayClock([stamp, stamp + timedelta(days=1)])
    assert replay.now() == stamp
    assert list(replay) == [stamp, stamp + timedelta(days=1)]
    assert replay.exhausted() is True
    assert replay.remaining() == 0
    assert replay.now() == stamp + timedelta(days=1)

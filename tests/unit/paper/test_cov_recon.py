"""Fail-closed branches of the paper reconciliation helpers.

All frames are SYNTHETIC correctness fixtures, never market evidence.
Malformed input raises; content mismatches are reported, not raised.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.paper.recon import (
    reconcile_broker_states,
    reconcile_equity,
    reconcile_fills,
)

T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _state(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "cash": 1000.0,
        "shares": {"AAPL": 3.0},
        "last_marks": {"AAPL": 100.0},
        "reject_count": 0,
        "halt_count": 0,
        "risk_gate_reject_count": 0,
        "n_fills": 1,
        "n_orders": 1,
        "kill_state": "ENABLED",
        "fill_convention": "next_open",
        "slot": "close",
    }
    base.update(over)
    return base


# -- reconcile_broker_states: malformed input fails closed --


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ([], {}),
        ({}, []),
        (None, {}),
        ({}, "state"),
    ],
)
def test_broker_states_require_dicts(expected: object, actual: object) -> None:
    with pytest.raises(ValueError, match="must be dicts"):
        reconcile_broker_states(expected, actual)  # type: ignore[arg-type]


@pytest.mark.parametrize("tol", [-1e-9, float("nan"), float("inf")])
def test_broker_states_reject_bad_tolerance(tol: float) -> None:
    with pytest.raises(ValueError, match="tol must be finite"):
        reconcile_broker_states(_state(), _state(), tol=tol)


def test_broker_states_cap_the_mismatch_list() -> None:
    expected = _state(shares={f"S{i:02d}": 1.0 for i in range(60)})
    actual = _state(shares={f"S{i:02d}": 2.0 for i in range(60)})
    report = reconcile_broker_states(expected, actual)
    assert report["match"] is False
    assert report["n_mismatches"] == 60
    assert len(report["mismatches"]) == 50


def test_broker_states_flag_non_numeric_counters() -> None:
    report = reconcile_broker_states(_state(reject_count="many"), _state(reject_count="few"))
    assert report["match"] is False
    assert any(item.startswith("reject_count:") for item in report["mismatches"])


def test_broker_states_equal_non_numeric_scalars_do_not_mismatch() -> None:
    report = reconcile_broker_states(_state(cash="n/a"), _state(cash="n/a"))
    assert report["match"] is True
    assert "cash_delta" not in report["deltas"]


def test_broker_states_flag_non_numeric_shares() -> None:
    report = reconcile_broker_states(
        _state(shares={"AAPL": 3.0}),
        _state(shares={"AAPL": "three"}),
    )
    assert report["match"] is False
    assert "shares[AAPL]: non-numeric" in report["mismatches"]
    assert "share_deltas" not in report["deltas"]


def test_broker_states_absent_share_key_is_flat() -> None:
    report = reconcile_broker_states(_state(shares={"AAPL": 0.0}), _state(shares={}))
    assert report["match"] is True
    assert report["deltas"]["cash_delta"] == 0.0


def test_broker_states_flag_shares_present_in_one_book_only() -> None:
    report = reconcile_broker_states(
        _state(shares={"AAPL": 3.0, "MSFT": 1.0}),
        _state(shares={"AAPL": 3.0}),
    )
    assert report["match"] is False
    assert any(item.startswith("shares[MSFT]:") for item in report["mismatches"])


def test_broker_states_flag_marks_present_in_one_book_only() -> None:
    report = reconcile_broker_states(
        _state(last_marks={"AAPL": 100.0, "MSFT": 10.0}),
        _state(last_marks={"AAPL": 100.0}),
    )
    assert report["match"] is False
    assert "last_marks[MSFT]: present in one book only" in report["mismatches"]


def test_broker_states_flag_non_numeric_marks() -> None:
    report = reconcile_broker_states(
        _state(last_marks={"AAPL": 100.0}),
        _state(last_marks={"AAPL": float("nan")}),
    )
    assert report["match"] is False
    assert "last_marks[AAPL]: non-numeric" in report["mismatches"]


def test_broker_states_flag_mark_drift() -> None:
    report = reconcile_broker_states(
        _state(last_marks={"AAPL": 100.0}),
        _state(last_marks={"AAPL": 101.0}),
        tol=1e-3,
    )
    assert report["match"] is False
    assert report["deltas"]["mark_deltas"] == {"AAPL": pytest.approx(1.0)}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kill_state", "HALT_NEW_ORDERS"),
        ("fill_convention", "same_bar"),
        ("slot", "open"),
    ],
)
def test_broker_states_flag_enum_drift(key: str, value: str) -> None:
    report = reconcile_broker_states(_state(), _state(**{key: value}))
    assert report["match"] is False
    assert any(item.startswith(f"{key}:") for item in report["mismatches"])


def test_broker_states_identical_books_match_with_deltas() -> None:
    report = reconcile_broker_states(_state(), _state())
    assert report["match"] is True
    assert report["n_mismatches"] == 0
    assert report["mismatches"] == []
    assert report["deltas"]["cash_delta"] == 0.0
    assert report["tol"] == 1e-9
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False


# -- reconcile_equity --


def _equity(rows: list[tuple[datetime, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [row[0] for row in rows],
            "nav": [row[1] for row in rows],
        }
    )


def test_equity_missing_columns_raise_for_either_side() -> None:
    good = _equity([(T0, 1.0)])
    with pytest.raises(ValueError, match="expected equity missing columns: \\['nav'\\]"):
        reconcile_equity(pl.DataFrame({"event_time": [T0]}), good)
    with pytest.raises(ValueError, match="actual equity missing columns"):
        reconcile_equity(good, pl.DataFrame({"nav": [1.0]}))
    with pytest.raises(ValueError, match=r"missing columns: \['ts'\]"):
        reconcile_equity(good, good, time_col="ts")


def test_equity_without_shared_timestamps_is_not_a_match() -> None:
    report = reconcile_equity(
        _equity([(T0, 1.0)]),
        _equity([(T0 + timedelta(days=1), 1.0)]),
    )
    assert report["status"] == "no_shared_timestamps"
    assert report["match"] is False
    assert report["n_matched"] == 0
    assert report["unmatched_expected"] == 1
    assert report["unmatched_actual"] == 1


def test_equity_null_nav_is_counted_not_treated_as_missing_row() -> None:
    report = reconcile_equity(_equity([(T0, 1.0)]), _equity([(T0, None)]))
    assert report["matched_null_nav"] == 1
    assert report["match"] is False
    assert report["unmatched_expected"] == 0
    assert report["unmatched_actual"] == 0
    assert report["status"] == "no_shared_timestamps"


def test_equity_last_delta_uses_the_latest_shared_timestamp() -> None:
    late = T0 + timedelta(minutes=5)
    report = reconcile_equity(
        _equity([(late, 100.0), (T0, 10.0)]),
        _equity([(T0, 10.0), (late, 102.0)]),
    )
    assert report["n_matched"] == 2
    assert report["last_nav_delta"] == pytest.approx(2.0)
    assert report["max_abs_nav_delta"] == pytest.approx(2.0)
    assert report["mean_nav_delta"] == pytest.approx(1.0)
    assert report["match"] is False


def test_equity_unmatched_timestamps_break_the_match() -> None:
    report = reconcile_equity(
        _equity([(T0, 1.0), (T0 + timedelta(minutes=1), 2.0)]),
        _equity([(T0, 1.0)]),
    )
    assert report["n_matched"] == 1
    assert report["unmatched_expected"] == 1
    assert report["match"] is False


def test_equity_exact_path_matches() -> None:
    rows = [(T0, 1.0), (T0 + timedelta(minutes=1), 2.0)]
    report = reconcile_equity(_equity(rows), _equity(rows))
    assert report["match"] is True
    assert report["max_abs_nav_delta"] == 0.0
    assert report["live_pnl_claim"] is False


# -- reconcile_fills --


def _fills(rows: list[tuple[str, datetime, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [row[0] for row in rows],
            "fill_time": [row[1] for row in rows],
            "quantity": [row[2] for row in rows],
            "price": [row[3] for row in rows],
        }
    )


def test_fills_missing_columns_raise_for_either_side() -> None:
    good = _fills([("AAPL", T0, 1.0, 10.0)])
    with pytest.raises(ValueError, match="expected fills missing columns"):
        reconcile_fills(pl.DataFrame({"security_id": ["AAPL"]}), good)
    with pytest.raises(ValueError, match="actual fills missing columns"):
        reconcile_fills(good, pl.DataFrame({"security_id": ["AAPL"]}))
    with pytest.raises(ValueError, match=r"missing columns: \['symbol'\]"):
        reconcile_fills(good, good, key=("symbol",))


def test_fills_duplicate_keys_are_reported_not_dropped() -> None:
    dup = _fills([("AAPL", T0, 1.0, 10.0), ("AAPL", T0, 1.0, 10.0)])
    report = reconcile_fills(dup, dup)
    assert report["match"] is False
    assert any("expected frame has 1 duplicated key rows" in m for m in report["mismatches"])
    assert any("actual frame has 1 duplicated key rows" in m for m in report["mismatches"])


def test_fills_unmatched_rows_break_the_match() -> None:
    report = reconcile_fills(
        _fills([("AAPL", T0, 1.0, 10.0), ("MSFT", T0, 2.0, 20.0)]),
        _fills([("AAPL", T0, 1.0, 10.0)]),
    )
    assert report["match"] is False
    assert report["unmatched_expected"] == 1
    assert report["unmatched_actual"] == 0
    assert any("unmatched fills" in m for m in report["mismatches"])


def test_fills_quantity_and_price_mismatches_are_counted() -> None:
    report = reconcile_fills(
        _fills([("AAPL", T0, 1.0, 10.0)]),
        _fills([("AAPL", T0, 2.0, 11.0)]),
    )
    assert report["n_qty_mismatch"] == 1
    assert report["n_price_mismatch"] == 1
    assert report["max_abs_qty_delta"] == pytest.approx(1.0)
    assert report["max_abs_price_delta"] == pytest.approx(1.0)
    assert report["n_matched"] == 1
    assert report["match"] is False


def test_fills_exact_match_reports_no_mismatch() -> None:
    rows = [("AAPL", T0, 1.0, 10.0)]
    report = reconcile_fills(_fills(rows), _fills(rows))
    assert report["match"] is True
    assert report["mismatches"] == []
    assert report["max_abs_qty_delta"] == 0.0
    assert report["max_abs_price_delta"] == 0.0
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False


def test_fills_empty_frames_match() -> None:
    empty = _fills([])
    report = reconcile_fills(empty, empty)
    assert report["match"] is True
    assert report["n_matched"] == 0

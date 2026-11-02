"""Quote age, clock skew, and disordered ticks halt before an order exists."""

from __future__ import annotations

import polars as pl
import pytest

from quant_fund.execution.quote_gate import (
    QuoteHalt,
    note_quote,
    require_fresh_quote,
    require_monotonic_event_times,
)
from quant_fund.paper.loop import _validate_bar_panel

_AGE = 5_000_000_000


def test_fresh_tick_and_exact_age_limit_are_usable() -> None:
    note_quote(symbol="AAPL", quote_ns=10, now_ns=10, max_age_ns=_AGE, previous_ns=None)
    note_quote(symbol="AAPL", quote_ns=0, now_ns=_AGE, max_age_ns=_AGE, previous_ns=None)
    require_fresh_quote(symbol="AAPL", quote_ns=10, now_ns=10 + _AGE, max_age_ns=_AGE)


def test_stale_skew_disorder_and_duplicate_ticks_halt() -> None:
    with pytest.raises(QuoteHalt, match="stale quote for AAPL"):
        note_quote(symbol="AAPL", quote_ns=0, now_ns=_AGE + 1, max_age_ns=_AGE, previous_ns=None)
    with pytest.raises(QuoteHalt, match="clock skew for AAPL"):
        note_quote(symbol="AAPL", quote_ns=11, now_ns=10, max_age_ns=_AGE, previous_ns=None)
    with pytest.raises(QuoteHalt, match="out-of-order quote for AAPL"):
        note_quote(symbol="AAPL", quote_ns=9, now_ns=12, max_age_ns=_AGE, previous_ns=10)
    with pytest.raises(QuoteHalt, match="duplicate quote for AAPL"):
        note_quote(symbol="AAPL", quote_ns=10, now_ns=10, max_age_ns=_AGE, previous_ns=10)


def test_unusable_clocks_halt() -> None:
    with pytest.raises(QuoteHalt, match="not usable"):
        note_quote(symbol="AAPL", quote_ns=-1, now_ns=10, max_age_ns=_AGE, previous_ns=None)
    with pytest.raises(QuoteHalt, match="not usable"):
        require_fresh_quote(symbol="AAPL", quote_ns=True, now_ns=10, max_age_ns=_AGE)  # type: ignore[arg-type]
    with pytest.raises(QuoteHalt, match="stale quote"):
        require_fresh_quote(symbol="MSFT", quote_ns=1, now_ns=1 + _AGE + 1, max_age_ns=_AGE)


def test_equal_event_times_are_one_cross_section() -> None:
    require_monotonic_event_times(["2024-01-03", "2024-01-03", "2024-01-04"])
    bars = pl.DataFrame(
        {
            "event_time": ["2024-01-03", "2024-01-03", "2024-01-04"],
            "security_id": ["A", "B", "A"],
        }
    )
    _validate_bar_panel(bars)


def test_backward_event_time_halts_before_any_order() -> None:
    with pytest.raises(ValueError, match="out-of-order market data; no order sent"):
        require_monotonic_event_times(["2024-01-04", "2024-01-03"])
    bars = pl.DataFrame(
        {
            "event_time": ["2024-01-04", "2024-01-03"],
            "security_id": ["A", "A"],
            "close": [10.0, 9.0],
        }
    )
    with pytest.raises(ValueError, match="no order sent"):
        _validate_bar_panel(bars)


def test_duplicate_bar_keys_keep_their_own_error() -> None:
    bars = pl.DataFrame({"event_time": ["2024-01-03", "2024-01-03"], "security_id": ["A", "A"]})
    with pytest.raises(ValueError, match="duplicate bars for event_time/security_id"):
        _validate_bar_panel(bars)

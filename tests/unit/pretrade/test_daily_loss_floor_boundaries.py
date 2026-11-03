"""Daily-loss floor when the session open is not a positive NAV.

A zero session start leaves the floor unset. A later positive start still
arms it. These rows check the pre-trade book, not a loss result.
"""

from __future__ import annotations

from tests.unit.pretrade.support import TS_NS, make_engine


def test_zero_session_start_leaves_the_daily_floor_unset() -> None:
    engine = make_engine(nav=1_000.0, cash=1_000.0, max_daily_loss_fraction=0.1)
    engine.update_account(
        nav=1_000.0,
        mark_ts_ns=TS_NS,
        session_start_nav=0.0,
        peak_nav=1_000.0,
    )
    assert engine.book.daily_floor == float("inf")

    engine.update_account(
        nav=1_000.0,
        mark_ts_ns=TS_NS,
        session_start_nav=1_000.0,
        peak_nav=1_000.0,
    )
    assert engine.book.daily_floor == 900.0

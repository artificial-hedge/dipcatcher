"""Tests for models/bakery_lock.py — the mutex check must not be vacuous."""

from __future__ import annotations


def test_enter_and_exit_are_distinct_steps() -> None:
    """A fused enter+exit made ``in_cs`` never overlap, so
    ``sum(in_cs) > 1`` was dead code. Entering is a distinct step now:
    after 4 steps thread 0 is inside but has not completed."""
    from quant_fund.models.bakery_lock import run_bakery

    m, order = run_bakery(2, [0, 0, 0, 0])
    assert m and order == []
    m, order = run_bakery(2, [0, 0, 0, 0, 0])
    assert m and order == [0]


def test_second_thread_waits_while_first_inside() -> None:
    """Thread 1's gate must stay blocked while thread 0 holds the CS —
    order is appended at exit, so it must still be empty mid-schedule."""
    from quant_fund.models.bakery_lock import run_bakery

    # thread 0: ticket -> gate -> enter CS; thread 1: ticket -> gate (blocked)
    m, order = run_bakery(2, [0, 0, 0, 0, 1, 1, 1])
    assert m and order == []
    m, order = run_bakery(2, [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1])
    assert m and order == [0, 1]


def test_bench_mutex_still_verified() -> None:
    from quant_fund.models.bakery_lock import bench_bakery_lock

    out = bench_bakery_lock()
    assert out["synthetic_mutual_exclusion"] == 1.0
    assert out["synthetic_no_double_entry"] == 1.0

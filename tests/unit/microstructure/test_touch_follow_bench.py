"""Tests for touch_follow — post-fill touch-state decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.touch_follow_bench import (
    _kernels,
    touch_follow_sim,
)


def _mk_book(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    bb = 100 + np.cumsum(rng.integers(-1, 2, n))
    ba = bb + 3 + rng.integers(0, 3, n)
    return (
        bb.astype(np.int64),
        ba.astype(np.int64),
        rng.integers(1, 20, n).astype(np.int64),
        rng.integers(1, 20, n).astype(np.int64),
    )


def test_kernels_instant_is_pure_emptying_when_touch_kept() -> None:
    # Flat quotes: no fill empties the touch, every instant is exactly 0.
    n = 400
    bb = np.full(n, 100, dtype=np.int64)
    ba = np.full(n, 104, dtype=np.int64)
    dep = np.full(n, 10, dtype=np.int64)
    fills = np.array([10, 30, 50, 70, 90], dtype=np.int64)
    signs = np.ones(5, dtype=np.int64)
    out = _kernels(bb, ba, dep.copy(), dep.copy(), fills, signs, lag=20, tick_units=1.0)
    assert out["touch_empty_rate"] == 0.0
    assert out["instant_given_kept_ticks"] == 0.0
    assert out["n_fills"] == 5
    # Mid decomposition identity: unhit + hit = k200.
    assert (
        pytest.approx(out["k200_unhit_ticks"] + out["k200_hit_ticks"], abs=1e-3)
        == out["k200_ticks"]
    )


def test_kernels_emptying_flag_and_revisit() -> None:
    # One buy fill empties the ask level and the quote never returns.
    n = 300
    bb = np.full(n, 100, dtype=np.int64)
    ba = np.full(n, 104, dtype=np.int64)
    fills = np.array([10], dtype=np.int64)
    signs = np.array([1], dtype=np.int64)
    ba[10:] = 106  # level emptied: ask steps up 2 ticks at the fill
    dep = np.full(n, 10, dtype=np.int64)
    out = _kernels(bb, ba, dep.copy(), dep.copy(), fills, signs, lag=10, tick_units=1.0)
    assert out["touch_empty_rate"] == 1.0
    assert out["instant_given_empty_ticks"] == pytest.approx(1.0)  # half of 2-tick gap
    assert out["old_touch_revisit_share"] == 0.0


def test_kernels_resite_lag_detected() -> None:
    n = 300
    bb = np.full(n, 100, dtype=np.int64)
    ba = np.full(n, 104, dtype=np.int64)
    fills = np.array([10], dtype=np.int64)
    signs = np.array([1], dtype=np.int64)
    bb[10 + 5 :] = 101  # bid steps up 5 events after the fill
    dep = np.full(n, 10, dtype=np.int64)
    out = _kernels(bb, ba, dep.copy(), dep.copy(), fills, signs, lag=30, tick_units=1.0)
    assert out["unhit_resite_lag_events"] == 5.0
    assert out["unhit_touch_depth_growth_k200"] == 0.0


def test_touch_follow_sim_shapes() -> None:
    out = touch_follow_sim(None, horizon=3000, seed=7)
    assert out["n_fills"] > 0
    assert out["instant_given_empty_ticks"] is not None
    assert out["k200_unhit_share"] is not None
    # The emptying mediates instant in the sim too.
    assert abs(out["instant_given_kept_ticks"]) < 0.15

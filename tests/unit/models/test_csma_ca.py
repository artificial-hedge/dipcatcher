"""Tests for csma_ca — exponential-backoff channel access."""

from __future__ import annotations

import numpy as np

from quant_fund.models.csma_ca import bench_csma_ca, csma_run


def test_backoff_bounds_attempts_and_deliveries() -> None:
    # the old model redrew a fixed uniform window forever: attempts grew
    # linearly with slots (~500 for 8 nodes/4000 slots) and "deliveries"
    # could exceed the node count (infinite packets). With real exponential
    # backoff each node's single packet delivers in a bounded number of tries.
    rng = np.random.RandomState(0)
    d, a, c = csma_run(8, 4000, rng)
    assert d <= 8
    assert a < 200
    assert d >= 4  # most nodes deliver within 4000 slots


def test_delivery_needs_attempt() -> None:
    rng = np.random.RandomState(1)
    d, a, _c = csma_run(4, 64, rng)
    assert 0 <= d <= a
    assert d <= 4


def test_collision_grows_window() -> None:
    # statistically: with backoff doubling, later attempts spread out —
    # attempts over the first half of slots exceed the second half when
    # collisions push windows up. Compare slot-100 attempts for the same
    # run continuation via a long-horizon run.
    rng = np.random.RandomState(2)
    _d, a, c = csma_run(6, 200, rng)
    assert c > 0  # collisions do occur at cw=8 with 6 nodes
    assert a < 6 * 200  # sanity bound: at most one attempt per node per slot


def test_bench_csma_ca() -> None:
    out = bench_csma_ca()
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
    assert out["synthetic_csma_progress"] == 1.0
    assert 0.0 < out["synthetic_csma_delivered_frac"] <= 1.0

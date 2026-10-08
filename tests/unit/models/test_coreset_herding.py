"""Tests for greedy herding coreset selection."""

from __future__ import annotations

import numpy as np

from quant_fund.models.coreset_herding import bench_coreset_herding


def test_herding_tracks_pool_mean() -> None:
    # Welling herding picks argmax <x, mu_pool - mean_sel>; with the sign
    # flipped it selected outliers away from the mean (anti-herding) and
    # the coreset mean drifted far from the pool mean.
    from quant_fund.models.coreset_herding import _herd_select

    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, (200, 8))
    x[:50] += 3.0
    sel = _herd_select(x, 20)
    assert len(sel) == 20
    assert len(set(sel)) == 20
    assert np.linalg.norm(x[sel].mean(0) - x.mean(0)) < 1.0


def test_herding_deterministic() -> None:
    from quant_fund.models.coreset_herding import _herd_select

    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, (100, 8))
    assert _herd_select(x, 10) == _herd_select(x, 10)


def test_bench_coreset_herding() -> None:
    out = bench_coreset_herding()
    for key in (
        "synthetic_herd_acc",
        "synthetic_herd_random_acc",
        "synthetic_herd_full_acc",
        "synthetic_herd_gain",
    ):
        assert np.isfinite(out[key])
    assert 0.0 <= out["synthetic_herd_acc"] <= 1.0

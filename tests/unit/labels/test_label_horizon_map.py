"""Tests for quant_fund.labels.label_horizon_map."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.labels.label_horizon_map import label_horizon_map, label_horizon_map_bench


def _flat(n: int = 500) -> np.ndarray:
    return np.full(n, 100.0)


def _trend(n: int = 500) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(np.full(n, 0.001)))


def test_grid_shape_and_keys():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, 800)))
    cells = label_horizon_map(close, barrier_widths=(0.01, 0.02), horizons=(5, 10))
    assert len(cells) == 4
    for c in cells:
        assert c["n_labeled"] + c["n_unobservable"] == c["n_events"]
        assert c["positive_rate"] is None or 0.0 <= c["positive_rate"] <= 1.0


def test_wider_barrier_raises_unobservable_free_counts():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, 600)))
    cells = label_horizon_map(close, barrier_widths=(0.001, 0.1), horizons=(10,))
    # tighter barrier touches more often -> labels resolve before horizon
    assert cells[0]["mean_bars_to_touch"] <= cells[1]["mean_bars_to_touch"]


def test_flat_tape_positive_rate_via_min_ret():
    cells = label_horizon_map(_flat(), barrier_widths=(0.01,), horizons=(5,), min_ret=1e-9)
    # flat path: |ret| < min_ret -> label 0 -> positive_rate 0
    assert cells[0]["positive_rate"] == 0.0


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        label_horizon_map(np.array([100.0, -1.0] + [100.0] * 20))
    with pytest.raises(ValueError):
        label_horizon_map(_trend(100), barrier_widths=(-0.1,), horizons=(5,))


def test_bench_sealed_and_deterministic():
    r1 = label_horizon_map_bench(n_bars=1200)
    r2 = label_horizon_map_bench(n_bars=1200)
    assert r1["schema"] == "label_horizon_map.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["live_pnl_claim"] is False
    assert r1 == r2

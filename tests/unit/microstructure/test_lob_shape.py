"""Tests for microstructure/lob_shape.py — book-depth profile."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.lob_shape import (
    LOB_SHAPE_SCHEMA,
    BookProfile,
    fit_shape_exponent,
    lob_shape_bench,
    sample_profile,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def _prof(depth: list[float], n: int = 10) -> BookProfile:
    d = np.asarray(depth, dtype=np.float64)
    return BookProfile(
        distance_ticks=np.arange(d.size, dtype=np.float64),
        bid_depth=d,
        ask_depth=d,
        n_snapshots=n,
    )


def test_fit_shape_exponent_linear_book() -> None:
    # d(i) = 2(i+1) → s = 1 exactly
    s = fit_shape_exponent(_prof([2, 4, 6, 8, 10, 12, 14, 16]))
    assert s == pytest.approx(1.0, abs=1e-9)


def test_fit_shape_exponent_flat_book() -> None:
    assert fit_shape_exponent(_prof([5.0] * 8)) == pytest.approx(0.0, abs=1e-9)


def test_fit_shape_exponent_empty() -> None:
    assert np.isnan(fit_shape_exponent(_prof([0.0] * 8, n=0)))


def test_sample_profile_deterministic() -> None:
    cfg = ZILobConfig(seed=5, init_depth=6, band=8, density_exponent=1.0)
    a = sample_profile(config=cfg, horizon=100.0)
    b = sample_profile(config=cfg, horizon=100.0)
    assert a.n_snapshots == b.n_snapshots and a.n_snapshots > 0
    np.testing.assert_array_equal(a.bid_depth, b.bid_depth)
    np.testing.assert_array_equal(a.ask_depth, b.ask_depth)


def test_bench_smoke_and_schema() -> None:
    out = lob_shape_bench(n_seeds=2, horizon=150.0)
    assert out["schema"] == LOB_SHAPE_SCHEMA
    for arm in ("triangular_calm", "flat_drift"):
        assert "shape_exponent" in out["arms"][arm]
        assert "peak_tick" in out["arms"][arm]
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64

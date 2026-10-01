"""Tests for quant_fund.native.warmup_spec."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.native import reference
from quant_fund.native.warmup_spec import (
    FLOOR_TOL,
    measure_warmup,
    seam_residual,
    warmup_spec,
    warmup_spec_bench,
)


def _series(n: int = 512, seed: int = 0) -> np.ndarray:
    return 100.0 + np.cumsum(np.random.default_rng(seed).standard_normal(n) * 0.4)


def test_rolling_mean_floors_at_window_minus_1():
    s = _series()
    w = 8
    # cumsum-difference rolling mean is not bit-exact, but floors instantly
    assert measure_warmup(lambda x: reference.rolling_mean(x, w), s, 256) is None
    k = measure_warmup(lambda x: reference.rolling_mean(x, w), s, 256, tol=FLOOR_TOL)
    assert k == w - 1


def test_simple_returns_needs_one_exact():
    s = _series()
    assert measure_warmup(reference.simple_returns, s, 256) == 1


def test_wealth_index_is_nonlocal():
    s = _series()
    fn = lambda x: reference.wealth_index(reference.simple_returns(x))  # noqa: E731
    assert measure_warmup(fn, s, 256, tol=FLOOR_TOL, max_warmup=255) is None


def test_ema_transient_decays_geometrically():
    s = _series()
    fn = lambda x: reference.ema(x, 12)  # noqa: E731
    r32 = seam_residual(fn, s, 256, 32)
    r64 = seam_residual(fn, s, 256, 64)
    r128 = seam_residual(fn, s, 256, 128)
    assert r32 > r64 > r128 > 0.0
    assert r64 < r32 * 0.1


def test_seam_residual_nan_semantics():
    s = _series(64, 1)
    fn = lambda x: reference.rolling_std(x, 5)  # noqa: E731
    assert seam_residual(fn, s, 8, 4) <= 1e-8
    assert seam_residual(fn, s, 8, 0) == np.inf  # NaN tail vs number


def test_measure_warmup_rejects_bad_geometry():
    s = _series(64)
    with pytest.raises(ValueError, match="seam"):
        measure_warmup(reference.simple_returns, s, 0)
    with pytest.raises(ValueError, match="warmup"):
        seam_residual(reference.simple_returns, s, 32, 40)


def test_spec_table_statuses():
    table = warmup_spec(
        {
            "rolling_mean_8": lambda s: reference.rolling_mean(s, 8),
            "simple_returns": reference.simple_returns,
            "wealth_index": lambda s: reference.wealth_index(reference.simple_returns(s)),
        },
        _series(256),
        128,
    )
    rm = table["rolling_mean_8"]
    assert rm["status"] == "floored"  # cumsum-difference is never bit-exact
    assert rm["warmup_at_floor"] == 7
    assert table["simple_returns"]["status"] == "exact"
    assert table["simple_returns"]["exact_warmup"] == 1
    assert table["wealth_index"]["status"] == "nonlocal"


def test_bench_sealed_and_deterministic():
    r1 = warmup_spec_bench()
    r2 = warmup_spec_bench()
    assert r1["schema"] == "warmup_spec.v1"
    assert r1["data_label"] == "SYNTHETIC"
    ops = r1["interpretation"]["ops"]
    assert ops["rolling_mean_20"]["status"] == "floored"
    assert ops["wealth_index"]["status"] == "nonlocal"
    assert ops["simple_returns"]["status"] == "exact"
    assert r1 == r2

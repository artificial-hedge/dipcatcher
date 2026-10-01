import numpy as np
import pytest

from quant_fund.models.fractional_coint import (
    bench_fractional_coint,
    fractional_coint,
    gph_d,
    synth_fcoint,
)


def test_bench_fractional_coint_passes():
    r = bench_fractional_coint()
    assert r["score"] == 1.0


def test_gph_rw_above_stationary():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(1500))
    ar = np.zeros(1500)
    for t in range(1, 1500):
        ar[t] = 0.5 * ar[t - 1] + rng.standard_normal()
    assert gph_d(rw) > gph_d(ar)


def test_residual_memory_lower():
    y, x, _, _ = synth_fcoint(seed=3)
    r = fractional_coint(y, x)
    assert r["d_resid"] < r["d_y"]


def test_rejects_mismatch():
    with pytest.raises(ValueError):
        fractional_coint(np.zeros(300), np.zeros(200))


def test_rejects_short():
    with pytest.raises(ValueError):
        gph_d(np.zeros(50))

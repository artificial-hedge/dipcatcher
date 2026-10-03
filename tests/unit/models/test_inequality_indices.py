"""Inequality/poverty index tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.inequality_indices import (
    atkinson,
    bench_inequality_indices,
    between_group_ge,
    fgt,
    generalized_entropy,
    gini,
    lorenz,
    theil_l,
    theil_t,
)


def test_gini_lognormal_closed_form():
    rng = np.random.default_rng(0)
    sig = 0.7
    y = rng.lognormal(0, sig, 40000)
    g_true = 2 * norm.cdf(sig / np.sqrt(2)) - 1
    assert abs(gini(y) - g_true) / g_true < 0.02


def test_gini_bounds():
    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-12)
    y = np.concatenate([np.zeros(99), [1000.0]])
    assert gini(y) == pytest.approx(0.99, abs=0.02)


def test_theil_lognormal():
    rng = np.random.default_rng(2)
    sig = 0.6
    y = rng.lognormal(0, sig, 40000)
    assert abs(theil_t(y) - 0.5 * sig * sig) / (0.5 * sig * sig) < 0.08
    assert theil_l(y) > 0


def test_atkinson_bounded():
    rng = np.random.default_rng(3)
    y = rng.lognormal(0, 0.8, 10000)
    a = atkinson(y, 0.5)
    assert 0 < a < 1
    assert a < atkinson(y, 1.5) or a <= 1


def test_fgt():
    rng = np.random.default_rng(4)
    y = rng.lognormal(0, 0.8, 10000)
    z = np.quantile(y, 0.4)
    assert fgt(y, z, 0.0) == pytest.approx(0.4, abs=0.02)
    assert 0 < fgt(y, z, 2.0) < fgt(y, z, 1.0) < fgt(y, z, 0.0)


def test_between_group():
    rng = np.random.default_rng(5)
    y = rng.lognormal(0, 0.5, 5000)
    g = (rng.random(5000) < 0.5).astype(int)
    assert between_group_ge(y, g, 1.0) >= 0


def test_lorenz_endpoints():
    rng = np.random.default_rng(6)
    p, ly = lorenz(rng.lognormal(0, 1, 5000))
    assert ly[0] == 0 and ly[-1] == pytest.approx(1.0)
    assert (np.diff(ly) >= -1e-12).all()
    assert p.size == ly.size


def test_fail_closed():
    with pytest.raises(ValueError):
        gini(np.array([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    with pytest.raises(ValueError):
        gini(np.zeros(20))
    with pytest.raises(ValueError):
        fgt(np.ones(10), z=-1.0)
    with pytest.raises(ValueError):
        generalized_entropy(np.ones(20) * -1)


def test_bench_passes():
    out = bench_inequality_indices()
    assert out["synthetic_gini_err"] < 0.02
    assert out["synthetic_theil_t_err"] < 0.05

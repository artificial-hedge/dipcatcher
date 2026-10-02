"""Dirichlet-multinomial module tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dirichlet_multinomial import (
    bench_dirichlet_multinomial,
    dm_loglik,
    dm_minka,
    dm_minka_newton,
    dm_mom,
)


def _synth(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.array([0.2, 0.3, 0.5])
    a0 = 5.0
    theta = rng.dirichlet(a0 * p, size=n)
    n_i = rng.poisson(15, size=n) + 1
    return np.vstack([rng.multinomial(int(n_i[i]), theta[i]) for i in range(n)])


def test_mom_recovers_overdispersion():
    counts = _synth()
    fit = dm_mom(counts)
    # true rho = 1/(1+5) ~ 0.167
    assert 0.05 < fit["rho"] < 0.35
    assert float(fit["a0"]) > 0


def test_minka_beats_mom_loglik():
    counts = _synth()
    mom = dm_mom(counts)
    mle = dm_minka(counts)
    assert float(mle["loglik"]) >= float(mom["loglik_mom"]) - 1e-9
    np.testing.assert_allclose(np.asarray(mle["p_hat"]).sum(), 1.0, atol=1e-8)


def test_minka_newton_agrees_with_fixed_point():
    counts = _synth(n=200)
    m1 = dm_minka(counts)
    m2 = dm_minka_newton(counts)
    assert abs(float(m1["loglik"]) - float(m2["loglik"])) < 0.02


def test_dm_loglik_finite_and_ordered():
    counts = _synth(n=50)
    good = dm_loglik(counts, np.array([1.0, 1.5, 2.5]))
    bad = dm_loglik(counts, np.array([20.0, 0.1, 0.1]))
    assert np.isfinite(good)
    assert good > bad


def test_input_validation():
    with pytest.raises(ValueError):
        dm_mom(np.array([[1]]))
    with pytest.raises(ValueError):
        dm_loglik(_synth(5), np.array([1.0, -1.0, 1.0]))


def test_bench_passes():
    out = bench_dirichlet_multinomial(seed=7)
    assert out["synthetic_rho_err_mle"] < 0.12
    assert out["synthetic_p_max_err"] < 0.12
    assert out["synthetic_mle_gain"] >= -1e-9
    assert out["synthetic_score"] == 1.0

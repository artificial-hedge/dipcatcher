"""Marginal-likelihood estimators (HM, GD, Chib, SD, TI)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.marginal_likelihood import (
    chib,
    gelfand_dey,
    harmonic_mean,
    savage_dickey,
    thermodynamic_integration,
)


def _conjugate(seed: int = 0):
    """Normal-normal conjugate model with closed-form log m."""
    rng = np.random.default_rng(seed)
    n = 40
    y = rng.normal(0.8, 1.0, n)
    s2, t2 = 1.0, 4.0
    p = n / s2 + 1 / t2
    yb = y.mean()
    syy = float((y**2).sum())
    true_lm = (
        -n / 2 * np.log(2 * np.pi * s2)
        - 0.5 * np.log(2 * np.pi * t2)
        + 0.5 * np.log(2 * np.pi / p)
        - 0.5 * (syy / s2 - (n * yb / s2) ** 2 / p)
    )
    post_sd = float(np.sqrt(1.0 / p))
    post_mean = float(post_sd**2 * n * yb / s2)
    return y, s2, t2, p, true_lm, post_mean, post_sd


def test_chib_exact_on_conjugate():
    y, s2, t2, p, true_lm, pm, ps = _conjugate()
    est = chib(
        float(stats.norm.logpdf(y, pm, np.sqrt(s2)).sum()),
        float(stats.norm.logpdf(pm, 0, np.sqrt(t2))),
        float(stats.norm.logpdf(pm, pm, ps)),
    )
    assert est == pytest.approx(true_lm, abs=1e-9)


def test_gelfand_dey_exact_when_h_equals_posterior():
    y, s2, t2, p, true_lm, pm, ps = _conjugate(1)
    rng = np.random.default_rng(2)
    draws = rng.normal(pm, ps, 2000)
    ll = np.array([stats.norm.logpdf(y, m, np.sqrt(s2)).sum() for m in draws])
    lp = np.array([stats.norm.logpdf(m, 0, np.sqrt(t2)) for m in draws])
    lh = np.array([stats.norm.logpdf(m, pm, ps) for m in draws])
    est = gelfand_dey(ll, lp, lh)
    assert est == pytest.approx(true_lm, abs=0.3)


def test_harmonic_mean_ordering():
    ll = np.array([-10.0, -11.0, -9.5, -10.2])
    est = harmonic_mean(ll)
    assert est < float(ll.mean())
    assert np.isfinite(est)


def test_savage_dickey_recovers_density_ratio():
    rng = np.random.default_rng(3)
    draws = rng.normal(1.0, 0.3, 8000)
    at = 1.0
    est = savage_dickey(draws, float(stats.norm.pdf(at, 0, 2)), at)
    true = float(np.log(stats.norm.pdf(at, 0, 2) / stats.norm.pdf(at, 1.0, 0.3)))
    assert est == pytest.approx(true, abs=0.15)


def test_thermodynamic_integration_flat_loglik():
    # constant log lik: integral of E[logL] over beta in
    # [0,1] equals the constant
    means = np.full(7, -42.0)
    betas = np.linspace(0, 1, 7) ** 5
    assert thermodynamic_integration(means, betas) == pytest.approx(-42.0)


def test_input_validation():
    with pytest.raises(ValueError):
        harmonic_mean(np.array([np.nan, -1.0]))
    with pytest.raises(ValueError):
        gelfand_dey(np.ones(4), np.ones(4), np.full(4, np.nan))

"""Tests for models/duration.py — Engle-Russell ACD."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.duration import acd_diagnostics, acd_fit, acd_simulate


def test_eacd_recovers_params() -> None:
    rng = np.random.default_rng(7)
    x = acd_simulate(0.05, 0.10, 0.85, 2000, rng)
    fit = acd_fit(x, dist="exp")
    assert fit["converged"] == 1.0
    assert fit["alpha"] == pytest.approx(0.10, abs=0.08)
    assert fit["beta"] == pytest.approx(0.85, abs=0.12)
    assert fit["persistence"] == pytest.approx(0.95, abs=0.15)


def test_residuals_whitened() -> None:
    rng = np.random.default_rng(3)
    x = acd_simulate(0.05, 0.15, 0.80, 3000, rng)
    fit = acd_fit(x, dist="exp")
    diag = acd_diagnostics(np.asarray(fit["resid"]))
    assert diag["mean_resid"] == pytest.approx(1.0, abs=0.08)
    # residuals should be far less autocorrelated than raw durations
    assert diag["lb_pvalue"] > 0.01


def test_raw_durations_autocorrelated() -> None:
    rng = np.random.default_rng(9)
    x = acd_simulate(0.05, 0.15, 0.80, 3000, rng)
    diag = acd_diagnostics(x)
    assert diag["lb_pvalue"] < 0.05  # clustering present pre-filter


def test_wacd_runs() -> None:
    rng = np.random.default_rng(5)
    x = acd_simulate(0.05, 0.10, 0.85, 1500, rng)
    fit = acd_fit(x, dist="weibull")
    assert "gamma" in fit
    assert fit["gamma"] > 0.05
    assert np.isfinite(np.asarray(fit["resid"])).all()


def test_simulate_fail_closed() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        acd_simulate(0.05, 0.6, 0.6, 100, rng)  # alpha+beta >= 1
    with pytest.raises(ValueError):
        acd_simulate(-0.1, 0.1, 0.8, 100, rng)
    with pytest.raises(ValueError):
        acd_simulate(0.05, 0.1, 0.8, 5, rng)


def test_fit_fail_closed() -> None:
    with pytest.raises(ValueError):
        acd_fit(np.ones(30))
    with pytest.raises(ValueError):
        acd_fit(-np.abs(np.random.default_rng(0).standard_normal(100)))
    with pytest.raises(ValueError):
        acd_fit(np.full(100, np.nan))
    with pytest.raises(ValueError):
        acd_fit(np.random.default_rng(0).exponential(1.0, 100), dist="bogus")


def test_nll_weibull_mean_one_scale() -> None:
    # eps must have mean 1 -> Weibull scale lam = 1/Gamma(1+1/gamma).
    import math as _m

    from quant_fund.models.duration import _nll_weibull

    x = np.array([0.8, 1.2, 0.9, 1.1, 1.0, 0.7])
    gamma = 2.0
    theta = np.array([1.0, 0.0, 0.0, gamma])
    # _psi_path anchors psi[0] = mean(x); alpha=beta=0 keeps later psi = omega.
    psi = np.concatenate([[x.mean()], np.ones(x.size - 1)])
    lam = 1.0 / _m.gamma(1.0 + 1.0 / gamma)
    z = x / psi / lam
    expected = -float(
        np.sum(
            np.log(gamma)
            - np.log(psi)
            - gamma * np.log(lam)
            + (gamma - 1.0) * np.log(x / psi)
            - z**gamma
        )
    )
    assert _nll_weibull(theta, x) == pytest.approx(expected, rel=1e-12)


def test_wacd_resid_mean_one_on_weibull_data() -> None:
    # Fit WACD on true mean-1 Weibull innovations; standardized residuals
    # keep mean ~1 only when the scale is 1/Gamma(1+1/gamma).
    import math as _m

    rng = np.random.default_rng(11)
    gamma_true = 2.0
    scale = 1.0 / _m.gamma(1.0 + 1.0 / gamma_true)
    n, burn = 3000, 200
    omega, alpha, beta = 0.05, 0.10, 0.85
    eps = rng.weibull(gamma_true, size=n + burn) * scale
    x = np.empty(n + burn)
    psi = omega / (1.0 - alpha - beta)
    for i in range(n + burn):
        psi = omega + alpha * (x[i - 1] if i else 0.0) + beta * psi
        x[i] = psi * eps[i]
    fit = acd_fit(x[burn:], dist="weibull")
    assert float(np.mean(fit["resid"])) == pytest.approx(1.0, abs=0.05)


def test_acd_fit_raises_on_infeasible_init() -> None:
    # alpha+beta >= 0.999 hits the 1e12 penalty plateau; the fit must fail
    # closed instead of returning the penalty value as a "converged" fit.
    rng = np.random.default_rng(13)
    x = acd_simulate(0.05, 0.10, 0.85, 500, rng)
    with pytest.raises(ValueError, match="ACD fit failed"):
        acd_fit(x, dist="exp", init=np.array([x.mean(), 0.9, 0.5]))

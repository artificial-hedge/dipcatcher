"""Tests for models/skew_t.py — Hansen (1994) skewed Student-t."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.skew_t import skew_t_cdf, skew_t_fit, skew_t_pdf, skew_t_ppf


def test_symmetric_when_lambda_zero() -> None:
    x = np.linspace(-5, 5, 101)
    pdf = skew_t_pdf(x, nu=6.0, lam=0.0)
    assert np.allclose(pdf, pdf[::-1], atol=1e-9)  # symmetric about 0
    assert abs(float(skew_t_cdf(np.array([0.0]), 6.0, 0.0)[0]) - 0.5) < 1e-6


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-40, 40, 8001)
    area = float(np.trapezoid(skew_t_pdf(x, nu=5.0, lam=-0.4), x))
    assert abs(area - 1.0) < 1e-3


def test_ppf_inverts_cdf() -> None:
    for p in (0.05, 0.3, 0.5, 0.8, 0.97):
        x = skew_t_ppf(p, nu=7.0, lam=0.3, mu=0.1, sigma=1.2)
        assert abs(float(skew_t_cdf(np.array([x]), 7.0, 0.3, 0.1, 1.2)[0]) - p) < 1e-6


def test_ppf_lambda_zero_is_unit_variance_student_t() -> None:
    """λ=0 Hansen law is Student-t scaled to variance 1, not a root find."""
    from scipy.stats import t as student_t

    nu = 6.0
    scale = math.sqrt(nu / (nu - 2.0))
    ps = np.array([0.01, 0.05, 0.25, 0.5, 0.8, 0.97, 0.999])
    got = skew_t_ppf(ps, nu=nu, lam=0.0)
    expect = student_t.ppf(ps, nu) / scale
    assert isinstance(got, np.ndarray)
    assert got == pytest.approx(expect, rel=1e-12, abs=1e-12)


def test_ppf_vector_matches_scalar() -> None:
    ps = np.array([0.05, 0.3, 0.5, 0.8, 0.97])
    vec = skew_t_ppf(ps, nu=7.0, lam=0.3, mu=0.1, sigma=1.2)
    scalar = np.array([skew_t_ppf(float(p), nu=7.0, lam=0.3, mu=0.1, sigma=1.2) for p in ps])
    assert vec == pytest.approx(scalar, rel=1e-15, abs=1e-15)


def test_ppf_matches_brentq_of_scipy_cdf() -> None:
    """Analytic quantile matches Brent inversion of the scipy.stats.t splice."""
    from scipy.optimize import brentq
    from scipy.special import gammaln
    from scipy.stats import t as student_t

    nu, lam, mu, sigma = 7.0, -0.45, 0.1, 1.2
    log_c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    c = float(np.exp(log_c))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = float(np.sqrt(1.0 + 3.0 * lam**2 - a**2))
    t_scale = math.sqrt(nu / (nu - 2.0))

    def cdf_at(x: float) -> float:
        z = (x - mu) / sigma
        if z < -a / b:
            arg = t_scale * (b * z + a) / (1.0 - lam)
            return (1.0 - lam) * float(student_t.cdf(arg, nu))
        arg = t_scale * (b * z + a) / (1.0 + lam)
        return (1.0 - lam) * 0.5 + (1.0 + lam) * (float(student_t.cdf(arg, nu)) - 0.5)

    for p in (0.01, 0.05, 0.3, 0.5, 0.8, 0.95, 0.99):
        root = brentq(
            lambda x, level=p: cdf_at(x) - level,
            mu - 80.0 * sigma,
            mu + 80.0 * sigma,
            xtol=1e-12,
        )
        got = float(skew_t_ppf(p, nu=nu, lam=lam, mu=mu, sigma=sigma))
        assert got == pytest.approx(root, abs=1e-8, rel=1e-8)


def test_fit_recovers_negative_skew() -> None:
    rng = np.random.default_rng(0)
    u = rng.random(1500)
    x = np.asarray(skew_t_ppf(u, nu=6.0, lam=-0.3), dtype=float)
    out = skew_t_fit(x)
    assert out["lam"] < 0.0
    assert 3.0 < out["nu"] < 30.0
    assert abs(out["mu"]) < 0.3


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        skew_t_pdf(np.array([0.0]), nu=1.5, lam=0.0)
    with pytest.raises(ValueError):
        skew_t_pdf(np.array([0.0]), nu=6.0, lam=0.0, sigma=0.0)
    with pytest.raises(ValueError):
        skew_t_ppf(1.5, nu=6.0, lam=0.0)

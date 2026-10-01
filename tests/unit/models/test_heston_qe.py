"""Tests for heston_qe — Andersen QE discretization."""

import numpy as np
import pytest

from quant_fund.models.heston_qe import (
    bench_heston_qe,
    heston_qe_path,
    qe_variance_step,
)


def test_nonnegative_under_feller_violation() -> None:
    # 2 kappa theta = 0.024 << sigma^2 = 0.25 — deep violation
    d = heston_qe_path(seed=1, t=300, sigma=0.5, theta=0.04, kappa=0.3, v0=0.02)
    assert np.min(np.asarray(d["v"])) >= 0.0


def test_conditional_moments_match() -> None:
    rng = np.random.default_rng(2)
    v0, dt, kappa, theta, sigma = 0.05, 0.1, 2.0, 0.06, 0.3
    vt = np.full(8000, v0)
    out = qe_variance_step(vt, dt, kappa, theta, sigma, rng)
    ek = np.exp(-kappa * dt)
    m_t = theta + (v0 - theta) * ek
    s_t = v0 * sigma**2 * ek * (1 - ek) / kappa + theta * sigma**2 * (1 - ek) ** 2 / (2 * kappa)
    assert abs(out.mean() - m_t) / m_t < 0.02
    assert abs(out.var() - s_t) / s_t < 0.08


def test_long_run_mean_tracks_theta() -> None:
    d = heston_qe_path(seed=3, t=1500, kappa=4.0, theta=0.05, sigma=0.3)
    v = np.asarray(d["v"])
    assert abs(v[300:].mean() - 0.05) / 0.05 < 0.25


def test_shapes() -> None:
    d = heston_qe_path(seed=4, t=100)
    assert np.asarray(d["v"]).shape == (101,)
    assert np.asarray(d["ln_s"]).shape == (101,)
    assert np.all(np.isfinite(np.asarray(d["ln_s"])))


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        heston_qe_path(kappa=-1.0)
    with pytest.raises(ValueError):
        heston_qe_path(rho=1.2)
    with pytest.raises(ValueError):
        heston_qe_path(t=5)


def test_determinism() -> None:
    a = heston_qe_path(seed=6, t=100)
    b = heston_qe_path(seed=6, t=100)
    np.testing.assert_array_equal(a["v"], b["v"])
    np.testing.assert_array_equal(a["ln_s"], b["ln_s"])


def test_bench_schema_and_score() -> None:
    r = bench_heston_qe()
    for k in ("mean_v", "stat_var", "var_v", "min_v", "m_err", "s_err", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0

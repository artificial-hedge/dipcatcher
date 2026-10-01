"""FKML generalized lambda distribution tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fkml import (
    bench_fkml,
    gld_density_support,
    gld_mom,
    gld_quantile,
    gld_simulate,
    gld_starship,
    gld_valid_pdf,
)


def test_quantile_function_shape():
    lam = np.array([0.0, 1.0, 0.5, 0.5])
    q = gld_quantile(np.array([0.0, 0.5, 1.0]), lam)
    np.testing.assert_allclose(q, [-1.0, 0.0, 1.0])


def test_support_finite_and_infinite():
    lam_pos = np.array([0.0, 1.0, 0.5, 0.5])
    lo, hi = gld_density_support(lam_pos)
    assert lo == -1.0 and hi == 1.0
    lam_neg = np.array([0.0, 1.0, -0.1, 0.5])
    lo, hi = gld_density_support(lam_neg)
    assert lo == float("-inf")


def test_mom_recovers_symmetric_gld():
    rng = np.random.default_rng(0)
    lam = np.array([1.0, 0.8, 0.2, 0.2])
    x = gld_simulate(lam, 2000, rng)
    fit = gld_mom(x)
    lam_h = np.asarray(fit["lam"])
    assert abs(lam_h[2] - 0.2) < 0.15
    assert abs(lam_h[3] - 0.2) < 0.15


def test_starship_recovers_location():
    rng = np.random.default_rng(1)
    lam = np.array([1.0, 0.8, 0.15, 0.4])
    x = gld_simulate(lam, 1500, rng)
    fit = gld_starship(x, n_grid=20)
    lam_h = np.asarray(fit["lam"])
    assert abs(lam_h[0] - 1.0) < 0.5
    assert gld_valid_pdf(lam_h)


def test_valid_pdf_check():
    assert gld_valid_pdf(np.array([0.0, 1.0, 0.5, 0.5]))
    # extreme shapes can violate monotonicity
    assert isinstance(gld_valid_pdf(np.array([0.0, 1.0, -0.4, -0.4])), bool)


def test_input_validation():
    with pytest.raises(ValueError):
        gld_mom(np.ones(10))
    with pytest.raises(ValueError):
        gld_quantile(np.array([0.5]), np.array([0.0, -1.0, 0.5, 0.5]))


def test_bench_passes():
    out = bench_fkml(seed=9)
    assert out["synthetic_mom_l3_err"] < 0.3
    assert out["synthetic_mom_l4_err"] < 0.3
    assert out["synthetic_starship_l1_err"] < 0.6
    assert out["synthetic_valid"] == 1.0
    assert out["synthetic_score"] == 1.0

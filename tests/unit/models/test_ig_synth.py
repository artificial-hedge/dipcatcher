"""Unit tests for quant_fund.models._ig_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._ig_synth import (
    fr_gauss_pair,
    gauss_fit,
    multinomial_pair,
    ou_stationary_sigma,
    poisson_nmf,
    simplex_ls,
)


def test_fr_gauss_pair_well_formed() -> None:
    (m1, s1), (m2, s2) = fr_gauss_pair()
    assert s1 > 0 and s2 > 0 and m1 != m2


def test_gauss_fit_deterministic() -> None:
    np.testing.assert_array_equal(gauss_fit(0), gauss_fit(0))
    assert abs(gauss_fit(0, n=5000).mean() - 2.0) < 0.02


def test_gauss_fit_rejects_empty() -> None:
    with pytest.raises(ValueError, match="n"):
        gauss_fit(0, n=0)


def test_simplex_ls_recoverable() -> None:
    A, b = simplex_ls(0)
    assert A.shape == (24, 24) and b.shape == (24,)
    # b is A @ (simplex point) + tiny noise: least-squares recovers ~x*
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    assert x_ls.min() > -0.02  # near-simplex
    assert abs(x_ls.sum() - 1.0) < 0.05


def test_simplex_ls_rejects_zero_dim() -> None:
    with pytest.raises(ValueError, match="d"):
        simplex_ls(0, d=0)


def test_poisson_nmf_deterministic_nonneg() -> None:
    V1 = poisson_nmf(0)
    V2 = poisson_nmf(0)
    np.testing.assert_array_equal(V1, V2)
    assert V1.shape == (30, 20)
    assert (V1 > 0).all()  # +1e-6 keeps strictly positive for Bregman


def test_poisson_nmf_rejects_zero_rank() -> None:
    # r=0 gives w@h == all-zeros -> constant 1e-6 matrix (vacuous target)
    with pytest.raises(ValueError, match="r"):
        poisson_nmf(0, r=0)
    with pytest.raises(ValueError, match="m"):
        poisson_nmf(0, m=0)


def test_multinomial_pair_is_valid() -> None:
    p, q = multinomial_pair()
    assert abs(p.sum() - 1.0) < 1e-12 and abs(q.sum() - 1.0) < 1e-12
    assert (p > 0).all() and (q > 0).all()
    assert not np.allclose(p, q)


def test_ou_stationary_sigma_positive() -> None:
    assert ou_stationary_sigma() > 0

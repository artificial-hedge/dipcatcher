"""Tests for factor_analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.factor_analysis import bench_factor_analysis, factor_analysis, varimax


def test_paf_recovers_two_blocks():
    rng = np.random.default_rng(0)
    n = 600
    lam = np.array([[0.8, 0.0], [0.7, 0.0], [0.75, 0.0], [0.0, 0.8], [0.0, 0.7], [0.0, 0.75]])
    psi = 1 - (lam**2).sum(1)
    f = rng.standard_normal((n, 2))
    x = f @ lam.T + rng.standard_normal((n, 6)) * np.sqrt(psi)
    r = np.corrcoef(x.T)
    out = factor_analysis(r, 2)
    l_hat = np.asarray(out["loadings"])
    # first three variables share a factor
    assert (
        np.corrcoef(l_hat[:3, 0], lam[:3, 0])[0, 1] ** 2 > 0.6
        or np.corrcoef(l_hat[:3, 1], lam[:3, 0])[0, 1] ** 2 > 0.6
    )


def test_varimax_simple_structure():
    rng = np.random.default_rng(1)
    lam = rng.standard_normal((8, 2))
    rot = varimax(lam)
    assert rot.shape == (8, 2)
    # orthogonal rotation preserves communalities
    assert np.allclose((rot**2).sum(1), (lam**2).sum(1))


def test_fail_closed_shape():
    with pytest.raises(ValueError):
        factor_analysis(np.ones((3, 4)), 2)


def test_bench():
    out = bench_factor_analysis()
    assert out["synthetic_score"] == 1.0

"""Adversarial probes for sparse."""

import numpy as np
import pytest

from quant_fund.models import sparse as sp


def _design(n: int = 120, p: int = 30, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true = rng.choice(p, k, replace=False)
    beta = np.zeros(p)
    beta[true] = rng.uniform(2.0, 4.0, k)
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return x, y, set(true.tolist())


def test_lasso_cd_rejects_zero_max_iter():
    x, y, _ = _design()
    with pytest.raises(ValueError, match="max_iter"):
        sp.lasso_cd(x, y, lam=0.1, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        sp.lasso_cd(x, y, lam=0.1, max_iter=-3)


def test_lasso_cd_n_iter_counts_iterations():
    x, y, _ = _design()
    out = sp.lasso_cd(x, y, lam=0.1, max_iter=500, tol=1e-12)
    assert out["n_iter"] >= 1.0
    assert out["n_iter"] <= 500.0


def test_omp_recovers_support():
    x, y, true = _design()
    out = sp.omp(x, y, k=8)
    sel = set(np.asarray(out["selected"], dtype=int).tolist())
    assert true <= sel


def test_lars_path_monotone_l1_endpoints():
    x, y, _ = _design()
    out = sp.lars_path(x, y)
    lam = np.asarray(out["lambdas"])
    assert lam.size >= 1
    assert np.all(np.diff(lam) > -1e-8)
    assert out["beta_path"].shape[0] == x.shape[1]


def test_lasso_cd_sparsifies_at_high_lambda():
    x, y, _ = _design()
    out = sp.lasso_cd(x, y, lam=1e3)
    assert out["n_nonzero"] == 0.0


def test_adaptive_lasso_recovers_support():
    x, y, true = _design()
    out = sp.adaptive_lasso(x, y, lam=0.05)
    sel = set(np.where(np.abs(np.asarray(out["beta"])) > 1e-8)[0].tolist())
    assert true <= sel


def test_ebic_selects_nonempty():
    x, y, true = _design()
    out = sp.ebic_select(x, y)
    assert out["n_selected"] >= 1.0
    sel = set(np.where(np.abs(np.asarray(out["beta"])) > 1e-8)[0].tolist())
    assert len(sel & true) >= 3

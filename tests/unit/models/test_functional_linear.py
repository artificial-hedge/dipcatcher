"""Adversarial probes for functional_linear."""

import numpy as np
import pytest

from quant_fund.models import functional_linear as fl


def test_smooth_lambda_changes_output():
    """The declared smoothing must actually do something: different
    lambdas must give different eigenfunctions."""
    x = fl.synth_fpca(n=80, t=60, seed=0)["X"]
    e0 = np.asarray(fl.fpca(x, n_components=3, smooth_lambda=0.0)["eigenfunctions"])
    e1 = np.asarray(fl.fpca(x, n_components=3, smooth_lambda=5.0)["eigenfunctions"])
    assert not np.allclose(e0, e1)


def test_large_lambda_smooths_eigenfunctions():
    """Roughness (total second-difference energy) of the leading
    eigenfunction must shrink as lambda grows."""
    x = fl.synth_fpca(n=80, t=60, seed=0)["X"]
    rough = {}
    for lam in (0.0, 50.0):
        e = np.asarray(fl.fpca(x, n_components=1, smooth_lambda=lam)["eigenfunctions"])
        rough[lam] = float(np.sum(np.diff(e[:, 0], n=2) ** 2))
    assert rough[50.0] < rough[0.0]


def test_functional_lm_rejects_nan_x():
    d = fl.synth_flm(n=80, seed=0)
    x = np.asarray(d["x"]).copy()
    x[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fl.functional_lm(d["Y"], x)


def test_basis_rejects_degenerate_args():
    grid = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="n_basis"):
        fl.bspline_basis(grid, n_basis=0)
    with pytest.raises(ValueError, match="degree"):
        fl.bspline_basis(grid, n_basis=5, degree=-1)


def test_variance_explained_bounded():
    x = fl.synth_fpca(n=80, t=60, seed=0)["X"]
    out = fl.fpca(x, n_components=5)
    ve = np.asarray(out["variance_explained"])
    assert np.all(ve <= 1.0 + 1e-9)
    assert np.all(ve >= 0.0)


def test_bench_smoke():
    out = fl.bench_functional_linear()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_eig1_congruence"] > 0.9

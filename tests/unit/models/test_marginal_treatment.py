"""Adversarial probes for marginal_treatment."""

import numpy as np
import pytest

from quant_fund.models import marginal_treatment as mt


def test_synth_rejects_rho_boundary():
    with pytest.raises(ValueError, match="rho"):
        mt.synth_mte(n=200, rho=1.0)
    with pytest.raises(ValueError, match="rho"):
        mt.synth_mte(n=200, rho=-1.3)


def test_mte_rejects_unbalanced_treat():
    d = mt.synth_mte(n=400, seed=3)
    y = np.asarray(d["y"])
    z = np.asarray(d["z"])
    x = np.asarray(d["x"])
    t_all = np.ones(400)
    with pytest.raises(ValueError, match="treated and"):
        mt.marginal_te(y, t_all, z, x)
    t_few = np.zeros(400)
    t_few[:5] = 1.0
    with pytest.raises(ValueError, match="treated and"):
        mt.marginal_te(y, t_few, z, x)


def test_mte_rejects_bad_params():
    d = mt.synth_mte(n=300, seed=1)
    y, t, z, x = d["y"], d["treat"], d["z"], d["x"]
    with pytest.raises(ValueError, match="bw"):
        mt.marginal_te(y, t, z, x, bw=0.0)
    with pytest.raises(ValueError, match="bw"):
        mt.marginal_te(y, t, z, x, bw=np.nan)
    with pytest.raises(ValueError, match="n_grid"):
        mt.marginal_te(y, t, z, x, n_grid=2)
    with pytest.raises(ValueError, match="n_grid"):
        mt.marginal_te(y, t, z, x, n_grid=3.5)


def test_mte_recovers_sign_of_true_beta():
    d = mt.synth_mte(n=3000, seed=11, rho=0.8, beta=0.6)
    out = mt.marginal_te(d["y"], d["treat"], d["z"], d["x"])
    assert out["ate_mte"] > 0.0
    # MTE-integrated ATE should be far closer to truth than naive diff
    assert abs(out["ate_mte"] - 0.6) < abs(out["naive_diff"] - 0.6)


def test_local_linear_slope_exact_on_linear():
    p = np.linspace(0, 1, 200)
    y = 3.0 + 2.0 * p
    grid = np.linspace(0.1, 0.9, 9)
    s = mt._local_linear_slope(y, p, grid, bw=0.2)
    np.testing.assert_allclose(s, np.full(9, 2.0), atol=1e-6)
    m = mt._local_linear_mean(y, p, grid, bw=0.2)
    np.testing.assert_allclose(m, 3.0 + 2.0 * grid, atol=1e-6)


def test_bench_smoke():
    out = mt.bench_marginal_treatment()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_detects"] == 1.0

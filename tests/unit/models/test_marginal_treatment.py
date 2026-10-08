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


def test_att_weight_is_treated_survival_not_density(monkeypatch):
    """omega_ATT(u) must be P(p >= u | D=1) (survival), not a kernel
    density of treated propensities. Verified end-to-end by injecting a
    known propensity law and MTE curve."""
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    # treated = low-z half -> treated propensities sit in the lower tail
    treat = (z < np.median(z)).astype(float)
    y = rng.standard_normal(n)

    # force propensity p_i = sigmoid(z_i): beta = [0, 0(x), 1(z)]
    monkeypatch.setattr(mt, "_irls_logit", lambda t, px: np.array([0.0, 0.0, 1.0]))

    captured = {}

    def fake_slope(y_, p_, grid, bw):
        captured["grid"] = grid
        return grid.copy()  # MTE(u) = u, known

    monkeypatch.setattr(mt, "_local_linear_slope", fake_slope)
    out = mt.marginal_te(y, treat, z, x, bw=0.15, n_grid=15)

    p = 1.0 / (1.0 + np.exp(-z))
    p_t = p[treat == 1.0]
    gr = captured["grid"]
    w = (p_t[:, None] >= gr[None, :]).mean(axis=0)
    w = w / w.sum()
    expected = float(gr @ w)
    assert abs(out["att_mte"] - expected) < 1e-9
    # density-weighted alternative must NOT coincide (guards regression)
    k = np.exp(-0.5 * ((gr[None, :] - p_t[:, None]) / 0.15) ** 2).mean(axis=0)
    k = k / k.sum()
    assert abs(out["att_mte"] - float(gr @ k)) > 1e-4

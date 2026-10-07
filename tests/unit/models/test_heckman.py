"""Adversarial probes for heckman."""

import numpy as np
import pytest

from quant_fund.models import heckman as hk


def _data(n: int = 500, seed: int = 0):
    d = hk.synth_heckman(n=n, seed=seed)
    return d["y"], d["x"], d["w"], d["s"]


def test_synth_rejects_rho_boundary():
    with pytest.raises(ValueError, match="rho"):
        hk.synth_heckman(n=200, rho=1.5)
    with pytest.raises(ValueError, match="rho"):
        hk.synth_heckman(n=200, rho=-1.0)


def test_ml_raises_when_every_restart_fails(monkeypatch):
    y, x, w, s = _data()
    monkeypatch.setattr(hk, "_heckman_nll", lambda *a: np.inf)
    with pytest.raises(ValueError, match="restart"):
        hk.heckman_ml(y, x, w, s, n_restarts=2)


def test_se_beta_shape_and_positivity():
    y, x, w, s = _data()
    out = hk.heckman_two_step(y, x, w, s)
    se = np.asarray(out["se_beta"])
    assert se.shape == (x.shape[1],)
    assert np.all(se >= 0.0)
    assert np.all(np.isfinite(se))


def test_se_beta_depends_on_lambda_column():
    """The sandwich must run on the [1, x, lam] design: mechanically
    recompute it and compare to the reported se."""
    y, x, w, s = _data()
    out = hk.heckman_two_step(y, x, w, s)
    sel = s == 1.0
    from scipy import linalg as la

    w1 = np.column_stack([np.ones(w.shape[0]), w])
    gamma = np.asarray(out["gamma"])
    lam = hk.inverse_mills(w1[sel] @ gamma)
    x1 = np.column_stack([np.ones(int(sel.sum())), x[sel], lam])
    resid = np.asarray(y[sel]) - x1 @ np.concatenate(
        [np.array([out["intercept"]]), np.asarray(out["beta"]), [out["lambda_coef"]]]
    )
    xt_xi = la.inv(x1.T @ x1)
    cov = xt_xi @ (x1.T @ (x1 * resid[:, None] ** 2)) @ xt_xi
    expected = np.sqrt(np.diag(cov))[1 : 1 + x.shape[1]]
    np.testing.assert_allclose(np.asarray(out["se_beta"]), expected, rtol=1e-8)


def test_bench_smoke():
    out = hk.bench_heckman()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_twostep_beats_ols"] == 1.0

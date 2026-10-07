"""Adversarial probes for pareto_nbd (BG/NBD)."""

import numpy as np
import pytest
from scipy.special import gammaln

from quant_fund.models import pareto_nbd as pn


def _ref_loglik(r, alpha, a, b, x, tx, T):
    """Independent transcription of FHL (2005) eq. 3."""
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)
    first = gammaln(r + x) - gammaln(r) + gammaln(a + b) + gammaln(b + x)
    first -= gammaln(b) + gammaln(a + b + x)
    first += r * np.log(alpha)
    t1 = -(r + x) * np.log(alpha + T)
    t2 = np.where(
        x > 0,
        np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + tx),
        -np.inf,
    )
    return float(np.sum(first + np.logaddexp(t1, t2)))


def test_loglik_matches_fhl_eq3():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, 60).astype(float)
    T = rng.uniform(20, 60, 60)
    tx = np.where(x > 0, np.minimum(T - 1, rng.uniform(1, 50, 60)), 0.0)
    tx = np.clip(tx, 0.0, T)
    ll = pn.bgnbd_loglik(1.5, 4.0, 0.8, 9.0, x, tx, T)
    assert ll == pytest.approx(_ref_loglik(1.5, 4.0, 0.8, 9.0, x, tx, T), rel=1e-9)


def test_loglik_consistent_with_p_alive():
    """eq. 11 denominator must equal 1 + second/first brace ratio of eq. 3."""
    r, alpha, a, b = 1.2, 5.0, 0.9, 7.0
    x = np.array([2.0, 3.0, 1.0])
    tx = np.array([10.0, 5.0, 30.0])
    T = np.array([40.0, 40.0, 40.0])
    t1 = -(r + x) * np.log(alpha + T)
    t2 = np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + tx)
    ratio = np.exp(t2 - t1)
    pa = pn.p_alive_bgnbd(r, alpha, a, b, x, tx, T)
    np.testing.assert_allclose(pa, 1.0 / (1.0 + ratio), rtol=1e-10)


def test_cbs_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        pn.bgnbd_loglik(1.0, 1.0, 1.0, 1.0, [np.inf], [1.0], [10.0])
    with pytest.raises(ValueError, match="finite"):
        pn.p_alive_bgnbd(1.0, 1.0, 1.0, 1.0, [1.0], [1.0], [np.nan])


def test_predictions_reject_nonpositive_params():
    x = np.array([1.0, 2.0])
    tx = np.array([5.0, 6.0])
    T = np.array([20.0, 20.0])
    with pytest.raises(ValueError, match="positive"):
        pn.bgnbd_expected_purchases(1.0, 1.0, -0.5, 1.0, x, tx, T, 5.0)
    with pytest.raises(ValueError, match="positive"):
        pn.p_alive_bgnbd(0.0, 1.0, 1.0, 1.0, x, tx, T)


def test_fit_skips_nan_restarts(monkeypatch):
    """A NaN first restart must not poison best tracking."""
    rng = np.random.default_rng(1)
    n = 40
    x = rng.poisson(2.0, n).astype(float)
    T = np.full(n, 30.0)
    tx = np.where(x > 0, rng.uniform(1, 29, n), 0.0)

    class _Res:
        def __init__(self, fun, x_):
            self.fun = fun
            self.x = np.asarray(x_, dtype=float)

    calls = []

    def fake_minimize(neg, x0, method=None, bounds=None, options=None):
        calls.append(1)
        fun = np.nan if len(calls) == 1 else float(neg(x0))
        return _Res(fun, x0)

    monkeypatch.setattr(pn, "minimize", fake_minimize)
    out = pn.fit_bgnbd(x, tx, T)
    assert np.isfinite(out["neg_ll"])


def test_bench_smoke():
    out = pn.bench_pnbd()
    assert out["synthetic_mse_model"] < out["synthetic_mse_naive"] * 1.15

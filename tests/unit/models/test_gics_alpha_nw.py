"""Probe: gics_alpha(nw_lags>0) must produce HAC-consistent alpha SEs."""

from __future__ import annotations

import numpy as np

from quant_fund.models.factor_models import gics_alpha


def test_nw_lags_changes_se_under_autocorrelation() -> None:
    rng = np.random.default_rng(7)
    t, n = 400, 3
    m = rng.normal(0.0, 1.0, t)
    # AR(1) residuals -> OLS se materially understates; NW se must differ.
    e = np.zeros(t)
    eps = rng.normal(0.0, 0.5, t)
    for k in range(1, t):
        e[k] = 0.9 * e[k - 1] + eps[k]
    r = np.column_stack([0.0 + 1.2 * m + e + rng.normal(0, 0.01, t) for _ in range(n)])
    ols_out = gics_alpha(r, m, nw_lags=0)
    nw_out = gics_alpha(r, m, nw_lags=6)
    # same point estimates, different t-stats
    assert np.allclose(ols_out["alpha"], nw_out["alpha"])
    assert not np.allclose(ols_out["alpha_t"], nw_out["alpha_t"])
    # under strong positive autocorrelation the HAC se should be LARGER
    # (|t| smaller) for the intercept under persistent e — accept either
    # direction but require a material difference.
    assert np.max(np.abs(ols_out["alpha_t"] - nw_out["alpha_t"])) > 0.01


def test_nw_sandwich_matches_closed_form() -> None:
    rng = np.random.default_rng(11)
    t, n = 120, 2
    m = rng.normal(0.0, 1.0, t)
    r = 0.3 + 0.9 * m[:, None] + rng.normal(0, 0.4, (t, n))
    lags = 4
    out = gics_alpha(r, m, nw_lags=lags)
    x = np.column_stack([np.ones(t), m])
    beta = np.linalg.lstsq(x, r, rcond=None)[0]
    e = r[:, 0] - x @ beta[:, 0]
    s = e[:, None] * x
    S = s.T @ s
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        S += w * (s[j:].T @ s[:-j] + s[:-j].T @ s[j:])
    xtx_inv = np.linalg.pinv(x.T @ x)
    cov = xtx_inv @ S @ xtx_inv
    se_expected = np.sqrt(max(cov[0, 0], 0.0))
    t_expected = beta[0, 0] / se_expected
    assert abs(out["alpha_t"][0] - t_expected) < 1e-9

"""Probes: interrupted_time_series HAC SEs must match the OLS scale under
iid errors (not be sqrt(T) too small); diff_in_diff must fail closed
when every cell is a singleton."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.causal_panel import diff_in_diff, interrupted_time_series


def test_hac_se_matches_ols_scale_iid() -> None:
    """Under iid residuals the HAC cov should approximate the OLS cov
    sigma2 (X'X)^-1 — within ~30%, not off by a factor of T."""
    rng = np.random.default_rng(4)
    t_len = 120
    y = 1.0 + 0.02 * np.arange(t_len) + rng.normal(scale=0.5, size=t_len)
    out = interrupted_time_series(y, tau=60, hac_lag=0)
    t = np.arange(t_len, dtype=float)
    post = (t >= 60).astype(float)
    x = np.column_stack([np.ones(t_len), t, post, np.maximum(t - 59.0, 0.0)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    s2 = float(resid @ resid) / (t_len - 4)
    ols_se = np.sqrt(s2 * np.diag(np.linalg.inv(x.T @ x)))
    hac_se = np.asarray(out["se"])
    ratio = hac_se / ols_se
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), f"ratio={ratio}"


def test_hac_se_widens_under_autocorrelation() -> None:
    rng = np.random.default_rng(4)
    t_len = 120
    e = np.zeros(t_len)
    z = rng.normal(scale=0.5, size=t_len)
    for i in range(1, t_len):
        e[i] = 0.7 * e[i - 1] + z[i]
    y = 1.0 + 0.02 * np.arange(t_len) + e
    hac = interrupted_time_series(y, tau=60, hac_lag=4)
    ols_like = interrupted_time_series(y, tau=60, hac_lag=0)
    assert float(np.mean(hac["se"])) > float(np.mean(ols_like["se"]))


def test_did_fails_closed_on_singleton_cells() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        diff_in_diff(y, d, p)

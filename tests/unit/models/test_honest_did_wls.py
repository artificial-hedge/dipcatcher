"""Probe: the pre-trend wLS fit must actually weight by 1/se^2, not 1/se^4."""

from __future__ import annotations

import numpy as np

from quant_fund.models.honest_did import honest_did_sd


def test_wls_weights_are_one_over_se_squared() -> None:
    # 3 pre-period betas; the middle one has a huge se so under 1/se^2 it is
    # nearly ignored, while under 1/se^4 it is ignored almost entirely.
    # With 1/se^2 weights the fit must match the closed-form WLS solution.
    rp = np.array([-3.0, -2.0, -1.0, 1.0])
    beta = np.array([0.10, 5.0, 0.12, 1.5])  # middle is a wild outlier
    se = np.array([0.05, 50.0, 0.05, 0.10])
    out = honest_did_sd(beta, se, rp, post_rp=1, mbar=0.0)

    # closed-form WLS with weights 1/se^2 on the pre rows
    w = 1.0 / se[:3] ** 2
    X = np.column_stack([np.ones(3), rp[:3]])
    wx = X * w[:, None]
    wy = beta[:3] * w
    coef = np.linalg.solve(X.T @ wx, X.T @ wy)
    expected_dev = np.abs(beta[:3] - X @ coef).max()
    assert abs(out["max_pre_dev"] - expected_dev) < 1e-9, (
        f"max_pre_dev {out['max_pre_dev']} != closed-form WLS dev {expected_dev}"
    )


def test_wls_outlier_nearly_ignored() -> None:
    # under 1/se^2 the wild middle point (se=50 vs 0.05) contributes ~1e-6
    # of the weight: fitted line stays near beta=0.11; under 1/se^4 it is
    # ignored entirely giving a similar line — the distinguishing case is
    # a MODERATE se where the two weightings disagree materially.
    rp = np.array([-4.0, -3.0, -2.0, -1.0, 1.0])
    beta = np.array([0.0, 0.30, 0.10, 0.10, 1.0])
    se = np.array([0.10, 0.30, 0.10, 0.10, 0.10])
    out = honest_did_sd(beta, se, rp, post_rp=1, mbar=0.0)
    w = 1.0 / se[:4] ** 2
    X = np.column_stack([np.ones(4), rp[:4]])
    coef = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (beta[:4] * w))
    expected = np.abs(beta[:4] - X @ coef).max()
    assert abs(out["max_pre_dev"] - expected) < 1e-9

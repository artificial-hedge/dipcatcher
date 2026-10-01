"""Tests for the OOF pinball quantile stacker."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.fusion.quantile_stack import cross_fitted_quantile_stack


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(0.0, 1.0, n)
    taus = np.array([0.1, 0.5, 0.9])
    # member 0 = true quantiles of N(0,1) + small noise; member 1 = junk
    q0 = np.stack(
        [norm.ppf(t) + rng.normal(0, 0.05, n) for t in taus], axis=1
    )  # (n, taus) per-member grid: build (n, members, taus)
    junk = rng.normal(0.0, 3.0, (n, len(taus)))
    member_q = np.stack([q0, junk], axis=1)
    folds = [
        (np.arange(0, 100), np.arange(100, 150)),
        (np.arange(0, 150), np.arange(150, 200)),
    ]
    return member_q, y, taus, folds


def test_stack_beats_junk_member() -> None:
    member_q, y, taus, folds = _fixture()
    out = cross_fitted_quantile_stack(member_q, y, taus, folds, alpha=0.5)
    assert out.oof_quantiles.shape == (200, 3)
    assert np.isnan(out.oof_quantiles[:100]).all()
    assert np.isfinite(out.oof_quantiles[100:]).all()
    # stacked pinball should be near the good member, far better than junk
    assert (out.oof_pinball < out.member_pinball[:, 1]).all()
    # monotone grid on every scored row
    assert (np.diff(out.oof_quantiles[100:], axis=1) >= -1e-9).all()


def test_monotone_repair_orders_rows() -> None:
    # members that disagree in ordering: member A low, member B high —
    # any repair must yield a non-decreasing grid.
    member_q, y, taus, folds = _fixture()
    member_q[:, 1, :] = member_q[:, 1, ::-1]  # reversed-grid junk member
    out = cross_fitted_quantile_stack(member_q, y, taus, folds, alpha=0.5)
    scored = out.fold_ids >= 0
    assert (np.diff(out.oof_quantiles[scored], axis=1) >= -1e-9).all()


def test_fold_contract_failures() -> None:
    member_q, y, taus, _ = _fixture()
    with pytest.raises(ValueError, match="overlap"):
        cross_fitted_quantile_stack(member_q, y, taus, [(np.arange(0, 100), np.arange(50, 150))])
    with pytest.raises(ValueError, match="exactly once"):
        cross_fitted_quantile_stack(
            member_q,
            y,
            taus,
            [
                (np.arange(0, 100), np.arange(100, 150)),
                (np.arange(0, 100), np.arange(140, 200)),
            ],
        )


def test_input_validation() -> None:
    member_q, y, taus, folds = _fixture()
    with pytest.raises(ValueError, match="strictly increasing"):
        cross_fitted_quantile_stack(member_q, y, np.array([0.5, 0.4, 0.9]), folds)
    bad = member_q.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cross_fitted_quantile_stack(bad, y, taus, folds)
    with pytest.raises(ValueError, match="positive"):
        cross_fitted_quantile_stack(member_q, y, taus, folds, alpha=0.0)

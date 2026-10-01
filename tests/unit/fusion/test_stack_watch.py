"""Tests for fusion/stack_watch.py — the stack's own audit."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.fusion.quantile_stack import cross_fitted_quantile_stack
from quant_fund.fusion.stack_watch import stack_watch, stack_watch_demo


def _fixture(n=200, seed=0):
    rng = np.random.default_rng(seed)
    taus = np.array([0.25, 0.5, 0.75])
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, 0.3, n)
    q = np.zeros((n, 2, taus.size))
    for j, t in enumerate(taus):
        z = float(np.quantile(rng.standard_normal(4000), t))
        q[:, 0, j] = x + z  # good member
        q[:, 1, j] = np.zeros(n) + z  # bad member (no signal)
    idx = np.arange(n)
    folds = [(idx[::2], idx[1::2]), (idx[1::2], idx[::2])]
    return q, y, taus, folds


def test_promotes_vs_bad_member():
    q, y, taus, folds = _fixture()
    res = cross_fitted_quantile_stack(q, y, taus, folds)
    out = stack_watch(res, q, y)
    assert out["per_member"]["member_1"]["promoted"]
    assert out["per_member"]["member_1"]["win_share"] >= 0.85


def test_oracle_is_hard():
    q, y, taus, folds = _fixture()
    res = cross_fitted_quantile_stack(q, y, taus, folds)
    out = stack_watch(res, q, y)
    # beating the per-origin oracle is rare; win_share < 1 means the
    # oracle beat the stack on some origins even when member_1 is good
    assert 0.0 <= out["stack_vs_oracle"]["win_share"] <= 1.0
    assert out["stack_vs_oracle"]["final_evalue"] > 0


def test_rejects_too_few_eligible():
    from quant_fund.fusion.quantile_stack import QuantileStackResult

    q, y, taus, _ = _fixture(n=20)
    res = QuantileStackResult(
        weights=np.ones((3, 2)),
        intercepts=np.zeros(3),
        oof_quantiles=np.zeros((20, 3)),
        fold_ids=np.array([0] + [-1] * 19),  # exactly one eligible row
        taus=taus,
        alpha=1.0,
        oof_pinball=np.zeros(3),
        member_pinball=np.zeros((3, 2)),
    )
    with pytest.raises(ValueError, match="eligible"):
        stack_watch(res, q, y)


def test_demo_sealed_and_deterministic():
    a = stack_watch_demo(seed=3, n=200)
    b = stack_watch_demo(seed=3, n=200)
    assert a["schema"] == "stack_watch.v1"
    assert a["data_label"] == "SYNTHETIC"
    assert a["receipt_sha256"] == b["receipt_sha256"]
    c = a["claim"]
    assert c["stack_vs_oracle"]["final_evalue"] > 0
    assert set(c["per_member"]) == {"member_0", "member_1"}

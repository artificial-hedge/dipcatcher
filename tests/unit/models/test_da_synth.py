"""Unit tests for quant_fund.models._da_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._da_synth import (
    W,
    finite_diff_grad,
    grad_corr,
    loss_np,
    sel_data,
    topk_mask,
)


def test_sel_data_deterministic() -> None:
    a = sel_data(4)
    b = sel_data(4)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_topk_mask_selects_largest() -> None:
    s = np.array([0.1, 0.9, -0.5, 0.3, 0.7, 0.2])
    m = topk_mask(s, k=2)
    assert m.sum() == 2.0
    assert m[1] == 1.0 and m[4] == 1.0


def test_loss_zero_at_target() -> None:
    # craft s so W @ (mask*s) == TARGET exactly: mask picks top-2 = dims {0,1}
    s = np.array([1.0, -1.0, -5.0, -5.0, -5.0, -5.0])
    assert np.array_equal(topk_mask(s), np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    # W @ (mask*s) = 1.5*1 + (-0.5)*(-1) = 2.0 = TARGET
    assert loss_np(s) == 0.0


def test_finite_diff_grad_matches_analytic() -> None:
    # mask={0,1} with a wide margin so eps perturbations cannot flip it:
    # W @ (mask*s) = 7.5 - 0.5 = 7.0 -> loss=(7-2)^2=25
    s = np.array([5.0, 1.0, -5.0, -5.0, -5.0, -5.0])
    g = finite_diff_grad(s)
    assert g[0] == pytest.approx(2 * 5 * W[0], abs=0.1)
    assert g[1] == pytest.approx(2 * 5 * W[1], abs=0.1)
    assert np.isfinite(g).all()


def test_grad_corr_degenerate() -> None:
    a = np.array([1.0, 0.0])
    assert grad_corr(a, a) == 1.0
    assert grad_corr(np.zeros(2), a) == 0.0
    assert grad_corr(a, -a) == -1.0

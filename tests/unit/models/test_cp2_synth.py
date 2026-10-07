"""Unit tests for quant_fund.models._cp2_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._cp2_synth import (
    cls_data,
    cov_width,
    cp2_data,
    fit_ridge,
    ridge_pred,
)


def test_cp2_data_splits_and_deterministic() -> None:
    a = cp2_data(5, n=100)
    b = cp2_data(5, n=100)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    assert a[0].shape == (100, 3) and a[2].shape == (100, 3)
    # train/test do not overlap rows
    assert not np.array_equal(a[0], a[2])


def test_ridge_fit_predicts() -> None:
    X, y, Xte, _yte = cp2_data(2, n=200)
    w = fit_ridge(X, y)
    pred = ridge_pred(w, Xte)
    assert pred.shape == (200,)
    assert np.isfinite(pred).all()


def test_cov_width_counts_and_measures() -> None:
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    y = np.array([0.5, 2.0])
    cov, wid = cov_width(lo, hi, y)
    assert cov == 0.5
    assert wid == 1.0


def test_cov_width_rejects_inverted_intervals() -> None:
    with pytest.raises(ValueError, match="lo <= hi"):
        cov_width(np.array([2.0]), np.array([1.0]), np.array([1.5]))


def test_cls_data_deterministic_binary() -> None:
    _X, y, _Xt, yt = cls_data(7, n=50)
    assert set(np.unique(y)) <= {0, 1}
    assert set(np.unique(yt)) <= {0, 1}

"""Tests for Hausdorff distances."""

from __future__ import annotations

import numpy as np

from quant_fund.models.hausdorff import (
    bench_hausdorff,
    directed_hausdorff,
    hausdorff,
    modified_hausdorff,
)


def test_hausdorff_self_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    assert hausdorff(X, X) == 0.0


def test_directed_asymmetry():
    A = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    B = np.array([[0.0, 0.5], [2.0, 0.5]])
    d_ab, _ = directed_hausdorff(A, B)
    d_ba, _ = directed_hausdorff(B, A)
    assert d_ab == max(d_ab, d_ba) == hausdorff(A, B)
    assert d_ab >= d_ba


def test_robust_outlier():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    Xo = np.vstack([X, [[100.0, 100.0]]])
    assert modified_hausdorff(Xo, X, 0.95) < hausdorff(X, Xo)


def test_bench_hausdorff():
    out = bench_hausdorff(seed=20261231)
    assert out["synthetic_hausdorff_self"] == 0.0
    assert all(np.isfinite(v) for v in out.values())

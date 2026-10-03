"""Tests for CP/Tucker tensor decomposition (models/tensor_decomp.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.tensor_decomp import (
    bench_tensor_decomp,
    core_consistency,
    cp_als,
    cp_missing_em,
    fold,
    synth_tensor,
    tucker_congruence,
    tucker_hooi,
    unfold,
)


@pytest.fixture
def panel():
    return synth_tensor(shape=(12, 8, 16), rank=3, noise=0.03, seed=5)


def test_unfold_fold_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5, 6))
    for m in range(3):
        a = unfold(x, m)
        assert a.shape == (x.shape[m], int(np.prod(x.shape) / x.shape[m]))
        back = fold(a, m, x.shape)
        assert np.allclose(back, x)


def test_cp_als_recovers_rank3(panel):
    out = cp_als(panel["X"], rank=3, n_iter=150, seed=0)
    assert float(out["rel_err"]) < 0.2
    assert out["fit"].shape == (12, 8, 16)


def test_cp_factor_congruence(panel):
    out = cp_als(panel["X"], rank=3, n_iter=150, seed=1)
    est = out["factors"]
    true_f = panel["true_factors"]
    for j in range(3):
        best = max(tucker_congruence(true_f[0][:, j], est[0][:, r]) for r in range(3))
        assert best > 0.9


def test_cp_determinism(panel):
    a = cp_als(panel["X"], rank=3, n_iter=50, seed=7)
    b = cp_als(panel["X"], rank=3, n_iter=50, seed=7)
    assert float(a["rel_err"]) == float(b["rel_err"])


def test_tucker_hooi_fit(panel):
    out = tucker_hooi(panel["X"], ranks=(3, 3, 3), n_iter=20)
    assert float(out["rel_err"]) < 0.2
    assert out["core"].shape == (3, 3, 3)
    for f, s in zip(out["factors"], (12, 8, 16), strict=True):
        assert f.shape == (s, 3)
        # orthonormal columns
        assert np.allclose(f.T @ f, np.eye(3), atol=1e-6)


def test_core_consistency_orders_rank(panel):
    cc3 = core_consistency(panel["X"], cp_als(panel["X"], 3, n_iter=100, seed=0)["factors"])
    cc5 = core_consistency(panel["X"], cp_als(panel["X"], 5, n_iter=60, seed=0)["factors"])
    assert cc3 > 80
    assert cc3 > cc5


def test_cp_missing_em(panel):
    rng = np.random.default_rng(9)
    mask = (rng.random(panel["X"].shape) > 0.2).astype(np.float64)
    out = cp_missing_em(panel["X"], mask, rank=3, n_iter=40, seed=0)
    assert float(out["miss_rmse"]) < 0.3
    assert math.isfinite(float(out["obs_rel_err"]))


def test_cp_validation():
    with pytest.raises(ValueError):
        cp_als(np.zeros((4, 4, 4)), rank=0)
    with pytest.raises(ValueError):
        cp_als(np.zeros((4, 4, 4)), rank=9)
    with pytest.raises(ValueError):
        cp_missing_em(np.zeros((4, 4, 4)), np.ones((4, 4, 3)), rank=2)
    with pytest.raises(ValueError):
        tucker_hooi(np.zeros((4, 4, 4)), ranks=(2, 2))


def test_tucker_congruence_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert tucker_congruence(a, a) == pytest.approx(1.0)
    assert tucker_congruence(a, -a) == pytest.approx(1.0)
    assert tucker_congruence(a, np.array([0.0, 0.0, 0.0])) == 0.0


def test_bench_keys():
    out = bench_tensor_decomp()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_cp_relerr"] < 0.2
    assert out["synthetic_factor_congruence"] > 0.9
    assert out["synthetic_core_consistency"] > 80
    assert out["synthetic_core_consistency"] > out["synthetic_core_consistency_overfit"]
    assert out["synthetic_determinism"] == 1.0

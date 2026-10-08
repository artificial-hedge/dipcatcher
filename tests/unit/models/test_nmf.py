"""Tests for nmf — Lee-Seung multiplicative updates."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.nmf import bench_nmf, cophenetic_correlation, nmf


def _planted(seed: int = 0, k: int = 3):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 2, (100, k))
    h = rng.uniform(0, 2, (k, 60))
    return w @ h, w, h


def test_reconstruction_r2():
    v, _, _ = _planted()
    out = nmf(v, 3, iters=400, seed=0)
    assert out["recon_r2"] > 0.99


def test_factors_nonnegative():
    v, _, _ = _planted()
    out = nmf(v, 3, iters=100, seed=0)
    assert np.all(np.asarray(out["w"]) >= 0)
    assert np.all(np.asarray(out["h"]) >= 0)


def test_kl_loss_works():
    v, _, _ = _planted()
    out = nmf(v, 3, loss="kl", iters=300, seed=0)
    assert out["recon_r2"] > 0.9


def test_deterministic():
    v, _, _ = _planted()
    a = nmf(v, 3, iters=100, seed=2)
    b = nmf(v, 3, iters=100, seed=2)
    assert np.allclose(a["w"], b["w"])


def test_cophenetic_stable_at_true_rank():
    v, _, _ = _planted(k=2)
    out = cophenetic_correlation(v, 2, n_runs=4, iters=150, seed=0)
    assert out["cophenetic_stab"] > 0.5


def test_fail_closed_negative_input():
    v = np.array([[1.0, -1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        nmf(v, 2)


def test_fail_closed_bad_rank():
    v = np.ones((10, 8))
    with pytest.raises(ValueError):
        nmf(v, 9)


def test_bench():
    out = bench_nmf()
    assert out["synthetic_score"] == 1.0

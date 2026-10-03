"""SSA decomposition tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.singular_spectrum import (
    bench_ssa,
    ssa_decompose,
    ssa_reconstruct,
    trajectory_matrix,
    w_correlation,
)


def test_trajectory_hankel():
    y = np.arange(20.0)
    X = trajectory_matrix(y, 4)
    assert X.shape == (4, 17)
    assert X[0, 0] == 0 and X[3, 16] == 19
    assert X[1, 0] == X[0, 1] == 1


def test_decompose_pure_sine_two_pairs():
    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 20)
    dec = ssa_decompose(y, 60)
    share = np.asarray(dec["share"])
    # a pure sinusoid concentrates in 2 eigentriples
    assert share[:2].sum() > 0.95


def test_reconstruct_recovers_signal():
    rng = np.random.default_rng(0)
    t = np.arange(150)
    y = np.sin(2 * np.pi * t / 15) + rng.normal(0, 0.2, 150)
    rec = ssa_reconstruct(y, 45, [[0, 1]])
    corr = np.corrcoef(rec["recon_0"], np.sin(2 * np.pi * t / 15))[0, 1]
    assert corr > 0.9


def test_residual_sums():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 100) + np.sin(np.arange(100) / 5)
    rec = ssa_reconstruct(y, 30, [[0], [1, 2]])
    tot = rec["recon_0"] + rec["recon_1"] + rec["residual"]
    assert np.allclose(tot, y, atol=1e-8)


def test_w_correlation_bounds():
    a = np.sin(np.linspace(0, 20, 100))
    b = np.cos(np.linspace(0, 20, 100))
    wc = w_correlation(a, b, 30)
    assert -1 <= wc <= 1


def test_bench_ssa():
    out = bench_ssa()
    assert out["synthetic_corr_sine"] > 0.85

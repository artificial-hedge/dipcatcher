"""Tests for regularized nonparametric IV (models/kernel_iv.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.kernel_iv import (
    bench_kernel_iv,
    conditional_moment_test,
    kernel_iv_estimate,
    landweber_fridman,
    synth_np_iv,
)


@pytest.fixture
def panel():
    d = synth_np_iv(n=300, strength=0.8, seed=3)
    return {k: np.asarray(v) for k, v in d.items()}


def test_synth_shapes(panel):
    assert panel["y"].shape == panel["x"].shape == panel["w"].shape


def test_kernel_iv_recovers(panel):
    est = kernel_iv_estimate(panel["y"], panel["x"], panel["w"], alpha=0.1)
    g_hat = np.asarray(est["g_hat"])
    g_true = np.asarray(panel["g_true"])
    scale = g_true @ g_hat / max(g_hat @ g_hat, 1e-9)
    rel = np.linalg.norm(scale * g_hat - g_true) / np.linalg.norm(g_true)
    assert rel < 0.6


def test_iv_beats_naive_smoothing(panel):
    # naive kernel smooth is biased under endogeneity
    est = kernel_iv_estimate(panel["y"], panel["x"], panel["w"], alpha=0.1)
    g_hat = np.asarray(est["g_hat"])
    g_true = np.asarray(panel["g_true"])
    scale = g_true @ g_hat / max(g_hat @ g_hat, 1e-9)
    iv_err = np.linalg.norm(scale * g_hat - g_true)
    x = panel["x"]
    bw = float(np.median(np.abs(x[:, None] - x[None, :])))
    kx = np.exp(-0.5 * ((x[:, None] - x[None, :]) / bw) ** 2)
    naive = np.linalg.solve(kx + 1e-2 * np.eye(x.size), kx @ panel["y"])
    assert iv_err < np.linalg.norm(naive - g_true)


def test_landweber_runs(panel):
    out = landweber_fridman(panel["y"], panel["x"], panel["w"])
    assert out["n_iter_run"] >= 1
    assert out["discrepancy"] >= 0.0


def test_conditional_moment(panel):
    est = kernel_iv_estimate(panel["y"], panel["x"], panel["w"], alpha=0.1)
    cmt = conditional_moment_test(panel["y"], panel["x"], panel["w"], np.asarray(est["g_hat"]))
    assert cmt["stat"] >= 0.0
    assert 0.0 <= cmt["p"] <= 1.0


def test_validation():
    with pytest.raises(ValueError):
        kernel_iv_estimate(np.ones(10), np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        kernel_iv_estimate(np.ones(40), np.ones(40), np.ones(30))
    with pytest.raises(ValueError):
        kernel_iv_estimate(np.ones(40), np.ones(40), np.ones(40), alpha=0.0)
    with pytest.raises(ValueError):
        landweber_fridman(np.ones(40), np.ones(40), np.ones(40), n_iter=0)


def test_determinism(panel):
    a = kernel_iv_estimate(panel["y"], panel["x"], panel["w"], alpha=0.1)["g_hat"]
    b = kernel_iv_estimate(panel["y"], panel["x"], panel["w"], alpha=0.1)["g_hat"]
    assert np.allclose(a, b)


def test_bench_keys():
    out = bench_kernel_iv()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_iv_beats_naive"] == 1.0
    assert out["synthetic_determinism"] == 1.0

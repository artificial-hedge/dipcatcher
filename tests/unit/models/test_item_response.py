"""Tests for item_response — Rasch/2PL joint MLE."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.item_response import (
    bench_item_response,
    rasch_jml,
    two_pl_jml,
)


def _sim(seed: int = 0, n_j: int = 200, n_i: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    th = rng.standard_normal(n_j)
    b = np.linspace(-1.5, 1.5, n_i)
    p = 1.0 / (1.0 + np.exp(-(th[:, None] - b[None, :])))
    return (rng.random((n_j, n_i)) < p).astype(float)


def test_rasch_recovers_difficulties():
    x = _sim()
    fit = rasch_jml(x)
    b_hat = fit["difficulty"]
    assert b_hat.shape == (x.shape[1],)
    assert np.abs(np.sort(b_hat) - np.linspace(-1.5, 1.5, x.shape[1])).max() < 0.9


def test_rasch_theta_ranks_subjects():
    rng = np.random.default_rng(1)
    th = rng.standard_normal(200)
    b = np.linspace(-1.5, 1.5, 8)
    p = 1.0 / (1.0 + np.exp(-(th[:, None] - b[None, :])))
    x = (rng.random(p.shape) < p).astype(float)
    th_hat = rasch_jml(x)["theta"]
    r1 = np.argsort(np.argsort(th))
    r2 = np.argsort(np.argsort(th_hat))
    assert np.corrcoef(r1.astype(float), r2.astype(float))[0, 1] > 0.6


def test_two_pl_shapes():
    x = _sim()
    fit = two_pl_jml(x)
    assert fit["theta"].shape == (x.shape[0],)
    assert fit["difficulty"].shape == (x.shape[1],)
    assert fit["discrimination"].shape == (x.shape[1],)
    assert (fit["discrimination"] > 0).all()


def test_fail_closed_nonbinary():
    with pytest.raises(ValueError):
        rasch_jml(np.array([[0.5, 1.0], [0.0, 1.0], [1.0, 0.0]]))


def test_bench():
    out = bench_item_response()
    assert out["synthetic_score"] == 1.0

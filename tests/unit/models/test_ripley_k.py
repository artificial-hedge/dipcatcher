"""Unit tests for quant_fund.models.ripley_k."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ripley_k import (
    bench_ripley_k,
    csr_envelope,
    ripley_k,
    ripley_l,
)

WINDOW = (0.0, 1.0, 0.0, 1.0)


def test_k_grows_with_r() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (150, 2))
    r = np.linspace(0.01, 0.15, 8)
    k = ripley_k(pts, WINDOW, r)
    assert np.all(np.diff(k) >= 0.0)
    assert np.all(k > 0.0)


def test_csr_l_near_r() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (200, 2))
    r = np.linspace(0.01, 0.15, 8)
    lv = ripley_l(ripley_k(pts, WINDOW, r))
    assert np.max(np.abs(lv - r)) < 0.05


def test_clustered_l_exceeds_r() -> None:
    rng = np.random.default_rng(2)
    parents = rng.uniform(0.1, 0.9, (15, 2))
    pts = np.concatenate([p + 0.03 * rng.standard_normal((15, 2)) for p in parents])
    pts = pts[(pts >= 0).all(axis=1) & (pts <= 1).all(axis=1)]
    r = np.linspace(0.01, 0.15, 8)
    lv = ripley_l(ripley_k(pts, WINDOW, r))
    assert np.max(lv - r) > 0.03


def test_csr_envelope_bounds() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, (200, 2))
    r = np.linspace(0.01, 0.15, 8)
    lo, hi = csr_envelope(200, WINDOW, r, n_sim=19, seed=0)
    lv = ripley_l(ripley_k(pts, WINDOW, r))
    assert np.all(lo <= hi)
    assert np.mean((lv >= lo) & (lv <= hi)) > 0.3


def test_rejects_bad_inputs() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (100, 2))
    r = np.linspace(0.01, 0.15, 8)
    with pytest.raises(ValueError):
        ripley_k(pts[:5], WINDOW, r)
    with pytest.raises(ValueError):
        ripley_k(pts, WINDOW, np.linspace(0.01, 0.9, 20))
    pts_out = pts.copy()
    pts_out[0, 0] = 2.0
    with pytest.raises(ValueError):
        ripley_k(pts_out, WINDOW, r)


def test_bench_ripley_k_score() -> None:
    out = bench_ripley_k()
    assert out["score"] == pytest.approx(1.0)
    assert out["synthetic_rk_csr_dev"] < 0.05

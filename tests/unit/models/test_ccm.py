"""Tests for models/ccm.py — Sugihara convergent cross-mapping (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ccm import ccm, simplex_project, takens_embed


def test_independent_series_low_rho() -> None:
    # Two independent white-noise series share no causal link: cross-map
    # skill must stay near zero. A prediction point inside its own library
    # self-matches at distance 0 and gets its own y back, which inflates rho
    # toward 1 — library and prediction sets are disjoint here.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(240)
    y = rng.standard_normal(240)
    out = ccm(x, y, dim=3, lib_sizes=np.array([30, 60, 90]), n_reps=10, seed=1)
    assert out["rho_max"] < 0.4


def test_driven_series_cross_maps() -> None:
    # y tracks an autocorrelated x with small noise: y's shadow manifold must
    # predict x (the CCM direction embeds the effect to recover the cause),
    # with skill rising as the library grows.
    rng = np.random.default_rng(2)
    n = 400
    e = rng.standard_normal(n)
    x = np.empty(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + e[t]
    y = 0.9 * x + 0.1 * rng.standard_normal(n)
    out = ccm(x, y, dim=3, lib_sizes=np.array([30, 60, 90, 120]), n_reps=8, seed=3)
    assert out["rho_max"] > 0.8
    assert out["convergent"] == 1.0


def test_ccm_deterministic_given_seed() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(160).cumsum()
    y = 0.5 * np.roll(x, 1) + rng.standard_normal(160)
    a = ccm(x, y, dim=3, n_reps=6, seed=9)
    b = ccm(x, y, dim=3, n_reps=6, seed=9)
    assert np.array_equal(a["rho"], b["rho"])


def test_simplex_project_basic() -> None:
    lib = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    lib_y = np.array([0.0, 1.0, 2.0, 3.0])
    pred = simplex_project(lib, lib_y, np.array([[1.0, 1.0]]), n_neighbors=1)
    assert pred[0] == pytest.approx(3.0)


def test_takens_embed_shape() -> None:
    x = np.arange(50, dtype=float) + 0.1 * np.sin(np.arange(50))
    m = takens_embed(x, dim=3, tau=2)
    assert m.shape == (50 - 4, 3)
    assert m[0, 0] == pytest.approx(x[4])

"""Unit tests for quant_fund.models.functional_pca."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.functional_pca import (
    bench_functional_pca,
    fpca,
    fpca_reconstruct,
)


def _bm_curves(n: int = 200, m: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, m)
    inc = rng.standard_normal((n, m)) * np.sqrt(grid[1] - grid[0])
    return np.cumsum(inc, axis=1), grid


def test_fpca_returns_keys() -> None:
    curves, grid = _bm_curves()
    out = fpca(curves, grid, n_components=3)
    for k in ("mean_curve", "eigenvalues", "eigenfunctions", "scores", "fve"):
        assert k in out
    assert np.asarray(out["eigenvalues"]).shape == (3,)
    assert np.asarray(out["scores"]).shape == (200, 3)


def test_eigenvalues_decay_and_match_bm() -> None:
    curves, grid = _bm_curves(n=400, m=200)
    out = fpca(curves, grid, n_components=3)
    lam = np.asarray(out["eigenvalues"])
    lam_true = np.array([1.0 / ((k - 0.5) ** 2 * np.pi**2) for k in range(1, 4)])
    assert np.all(np.diff(lam) < 0.0)
    assert np.linalg.norm(lam - lam_true) / np.linalg.norm(lam_true) < 0.3


def test_fve_first_component_bm() -> None:
    curves, grid = _bm_curves(n=400, m=200)
    out = fpca(curves, grid, n_components=3)
    fve = np.asarray(out["fve"])
    assert 0.7 < fve[0] < 0.92


def test_reconstruction_improves_with_components() -> None:
    curves, grid = _bm_curves(n=100, m=100)
    out = fpca(curves, grid, n_components=5)
    rec3 = fpca_reconstruct(out, 3)
    rec5 = fpca_reconstruct(out, 5)
    e3 = np.mean((curves - rec3) ** 2)
    e5 = np.mean((curves - rec5) ** 2)
    assert e5 < e3


def test_rejects_bad_input() -> None:
    curves, grid = _bm_curves(n=3)
    with pytest.raises(ValueError):
        fpca(curves, grid)
    curves2, _ = _bm_curves(n=50)
    curves2[0, 0] = np.nan
    with pytest.raises(ValueError):
        fpca(curves2, grid)


def test_bench_functional_pca_score() -> None:
    out = bench_functional_pca()
    assert out["score"] == pytest.approx(1.0)
    assert out["synthetic_fpca_lam_err"] < 0.25

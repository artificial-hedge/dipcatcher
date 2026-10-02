"""MDS tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mds import bench_mds, classical_mds, smacof_mds


def _euclid_d(seed: int = 0, n: int = 20):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    diff = x[:, None, :] - x[None, :, :]
    return x, np.sqrt((diff * diff).sum(axis=2))


def test_classical_recovers_distances():
    _, d = _euclid_d()
    out = classical_mds(d, 2)
    x = np.asarray(out["config"])
    dh = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(d.shape[0], 1)
    assert np.abs(d[iu] - dh[iu]).max() / d.max() < 1e-6
    assert out["gof"] > 0.99


def test_classical_eigenvalues_descending():
    _, d = _euclid_d()
    ev = np.asarray(classical_mds(d, 3)["eigenvalues"])
    assert (np.diff(ev) <= 1e-10).all()


def test_smacof_reduces_stress():
    _, d = _euclid_d()
    rng = np.random.default_rng(0)
    d_warp = d + 0.05 * np.abs(rng.normal(0, 1, d.shape))
    np.fill_diagonal(d_warp, 0.0)
    d_warp = (d_warp + d_warp.T) / 2
    out = smacof_mds(d_warp, 2, n_iter=100, seed=0)
    hist = np.asarray(out["stress_hist"])
    assert hist[-1] <= hist[0] + 1e-9
    assert out["stress"] < 0.1


def test_smacof_on_exact_dist_zero_stress():
    _, d = _euclid_d(3)
    out = smacof_mds(d, 2, n_iter=50, seed=0)
    assert out["stress"] < 0.02


def test_input_validation():
    with pytest.raises(ValueError):
        classical_mds(np.ones((3, 4)), 2)
    with pytest.raises(ValueError):
        classical_mds(-np.ones((6, 6)), 2)
    with pytest.raises(ValueError):
        smacof_mds(np.zeros((5, 5)), 5)


def test_bench_passes():
    out = bench_mds()
    assert out["synthetic_classical_dist_err"] < 1e-6
    assert out["synthetic_classical_gof"] > 0.99
    assert out["synthetic_smacof_stress"] < 0.02
    assert out["synthetic_score"] == 1.0

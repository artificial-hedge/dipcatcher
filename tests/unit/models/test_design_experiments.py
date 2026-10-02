"""Design of experiments canon."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.design_experiments import (
    box_behnken,
    central_composite,
    d_optimal,
    full_factorial_2k,
    latin_hypercube,
    plackett_burman,
)


def test_full_factorial_2k_orthogonal():
    f = full_factorial_2k(3)
    assert f.shape == (8, 3)
    xtx = f.T @ f
    assert np.allclose(np.diag(np.diag(xtx)), xtx)
    assert set(np.unique(f)) == {-1, 1}


def test_plackett_burman_orthogonal():
    pb = plackett_burman(12)
    assert pb.shape == (12, 11)
    xtx = pb.T @ pb
    assert np.allclose(np.diag(np.diag(xtx)), xtx, atol=1e-10)


def test_pb_rejects_unknown_size():
    with pytest.raises(ValueError):
        plackett_burman(16)


def test_ccd_alpha_rotatable():
    ccd = central_composite(2)
    assert np.abs(np.abs(ccd[4]).max() - np.sqrt(2)) < 1e-9


def test_box_behnken_shape():
    bb = box_behnken(3, n_center=3)
    assert bb.shape == (15, 3)
    # every row has exactly two nonzero entries or is a center
    nz = (bb != 0).sum(axis=1)
    assert set(np.unique(nz)) <= {0, 2}


def test_lhs_stratified_margins():
    x = latin_hypercube(40, 3, seed=1)
    assert x.shape == (40, 3)
    for j in range(3):
        col = np.sort(x[:, j])
        # every stratum covered
        assert col[0] < 0.1
        assert col[-1] > 0.9
        assert np.all(np.diff(col) < 0.2)


def test_d_optimal_beats_random():
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, (150, 2))
    cand = np.c_[np.ones(150), u, u[:, 0] * u[:, 1], u**2]
    dopt = d_optimal(cand, 10, seed=0)
    m = np.asarray(dopt["design"])
    _, ld_opt = np.linalg.slogdet(m.T @ m)
    s = rng.choice(150, 10, replace=False)
    _, ld_rand = np.linalg.slogdet(cand[s].T @ cand[s] + 1e-12 * np.eye(6))
    assert ld_opt > ld_rand

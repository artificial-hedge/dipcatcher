"""Tests for discrete Fréchet distance."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from quant_fund.models.frechet import bench_frechet, frechet_dist, frechet_path


def test_frechet_identical():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    d, _ = frechet_dist(P, P)
    assert d == 0.0


def test_frechet_known():
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    B = np.array([[0.0, 0.0], [1.0, 2.0]])
    d, _ = frechet_dist(A, B)
    assert abs(d - 2.0) < 1e-12


def test_frechet_path_monotone():
    A = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    B = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    path = frechet_path(A, B)
    for (i0, j0), (i1, j1) in pairwise(path):
        assert i1 >= i0 and j1 >= j0


def test_bench_frechet():
    out = bench_frechet(seed=20261231)
    assert out["synthetic_frechet_self"] == 0.0
    assert out["synthetic_frechet_shift_err"] < 1e-9
    assert all(np.isfinite(v) for v in out.values())

"""Tests for thin_plate."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.thin_plate import bench_thin_plate, thin_plate, thin_plate_predict


def test_interpolation_exact():
    rng = np.random.default_rng(0)
    xy = rng.random((40, 2))
    z = xy[:, 0] ** 2 - xy[:, 1]
    fit = thin_plate(xy, z, lam=0.0)
    assert np.abs(thin_plate_predict(fit, xy) - z).max() < 1e-4


def test_smoothing_beats_noise():
    rng = np.random.default_rng(1)
    xy = rng.random((80, 2)) * 3
    z = np.sin(xy[:, 0]) + np.cos(xy[:, 1])
    y = z + 0.1 * rng.standard_normal(80)
    fit = thin_plate(xy, y, lam=1e-3)
    pred = thin_plate_predict(fit, xy)
    assert np.mean((pred - z) ** 2) < np.mean((y - z) ** 2)


def test_fail_closed_duplicates():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        thin_plate(xy, np.ones(4))


def test_bench():
    out = bench_thin_plate()
    assert out["synthetic_score"] == 1.0

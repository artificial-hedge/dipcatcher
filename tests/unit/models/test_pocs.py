"""POCS / Dykstra / Douglas-Rachford tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.pocs import (
    alternating_projections,
    bench_pocs,
    douglas_rachford,
    dykstra,
    proj_affine,
    proj_ball,
    proj_box,
    proj_halfspace,
)


def test_proj_affine_idempotent():
    p = proj_affine(np.array([1.0, 0.0, 0.0]), 2.0)
    x = np.array([5.0, 1.0, -1.0])
    y = p(x)
    np.testing.assert_allclose(y, [2.0, 1.0, -1.0])
    np.testing.assert_allclose(p(y), y)


def test_proj_ball_inside_outside():
    p = proj_ball(np.zeros(3), 1.0)
    inside = np.array([0.3, 0.0, 0.0])
    np.testing.assert_allclose(p(inside), inside)
    outside = np.array([3.0, 0.0, 0.0])
    np.testing.assert_allclose(p(outside), [1.0, 0.0, 0.0])


def test_proj_halfspace_and_box():
    hs = proj_halfspace(np.array([1.0, 0.0]), 1.0)
    np.testing.assert_allclose(hs(np.array([5.0, 2.0])), [1.0, 2.0])
    np.testing.assert_allclose(hs(np.array([0.5, 2.0])), [0.5, 2.0])
    bx = proj_box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(bx(np.array([3.0, -2.0])), [1.0, 0.0])


def test_dykstra_finds_closest_point():
    # x0 projected onto line x1=0 vs ball: known answer
    a = np.array([1.0, 0.0])
    x0 = np.array([4.0, 0.5])
    projs = [proj_affine(a, 0.0), proj_ball(np.zeros(2), 1.0)]
    out = dykstra(x0, projs)
    np.testing.assert_allclose(np.asarray(out["x"]), [0.0, 0.5], atol=1e-6)


def test_alternating_projections_feasible():
    projs = [
        proj_affine(np.array([1.0, 1.0]), 1.0),
        proj_box(np.array([0.0, 0.0]), np.array([2.0, 0.4])),
    ]
    out = alternating_projections(np.array([5.0, 5.0]), projs)
    x = np.asarray(out["x"])
    assert x.sum() == pytest.approx(1.0, abs=1e-4)
    assert x[1] <= 0.4 + 1e-6


def test_douglas_rachford_feasible():
    p1 = proj_affine(np.array([1.0, 0.0]), 1.0)
    p2 = proj_ball(np.zeros(2), 2.0)
    out = douglas_rachford(np.array([5.0, 5.0]), p1, p2)
    x = np.asarray(out["x"])
    assert abs(x[0] - 1.0) < 1e-4
    assert np.linalg.norm(x) <= 2.0 + 1e-4


def test_input_validation():
    with pytest.raises(ValueError):
        proj_affine(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        proj_ball(np.zeros(2), -1.0)
    with pytest.raises(ValueError):
        dykstra(np.zeros(2), [])


def test_bench_passes():
    out = bench_pocs(seed=17)
    assert out["synthetic_dykstra_err"] < 1e-6
    assert out["synthetic_ap_resid"] < 1e-6
    assert out["synthetic_dr_resid"] < 1e-4
    assert out["synthetic_score"] == 1.0

"""Wave-314 geometry-processing module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.catmull_clark import cc_step, cube_mesh
from quant_fund.models.half_edge import HalfEdgeMesh
from quant_fund.models.laplacian_smooth import cotangent_laplacian, umbrella_smooth
from quant_fund.models.loop_subdiv import icosahedron, loop_step
from quant_fund.models.marching_cubes import marching_squares_contour
from quant_fund.models.nurbs_eval import bspline_basis, make_clamped_knots, nurbs_point


def test_nurbs_basis_partition() -> None:
    k = make_clamped_knots(6, 3)
    for u in [0.0, 0.3, 0.77, 0.999]:
        s = sum(bspline_basis(i, 3, u, k) for i in range(6))
        assert abs(s - 1.0) < 1e-10


def test_nurbs_endpoint() -> None:
    ctrl = np.array([[0, 0], [1, 2], [2, 0]])
    w = np.ones(3)
    k = make_clamped_knots(3, 2)
    assert np.allclose(nurbs_point(1.0, ctrl, w, k, 2), ctrl[-1])


def test_cc_counts() -> None:
    v, f = cube_mesh()
    v1, f1 = cc_step(v, f)
    assert len(v1) == 8 + 6 + 12 and len(f1) == 24


def test_loop_counts() -> None:
    v, f = icosahedron()
    v1, f1 = loop_step(v, f)
    assert len(v1) == 42 and len(f1) == 80


def test_half_edge_boundary() -> None:
    m = HalfEdgeMesh([[0, 1, 2], [2, 1, 3]])
    assert len(m.boundary_edges()) == 4
    assert m.face_neighbors(0) == {1}


def test_msquares_circle() -> None:
    ax = np.linspace(-1.5, 1.5, 16)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    f = np.sqrt(x**2 + y**2) - 1.0
    segs = marching_squares_contour(f, 0.0, ax[1] - ax[0])
    assert 20 < len(segs) < 200


def test_cot_laplacian_flat() -> None:
    v = np.array([[x, y, 0.0] for x in range(3) for y in range(3)])
    f = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]]
    assert np.linalg.norm(cotangent_laplacian(v, f, 4)) < 1e-10


def test_umbrella_shrinks() -> None:
    v = np.array([[1.0, 0], [0, 1.0], [-1.0, 0], [0, -1.0]])
    adj = [{1, 3}, {0, 2}, {1, 3}, {0, 2}]
    s = umbrella_smooth(v, adj, 0.5, 1)
    assert np.allclose(np.linalg.norm(s, axis=1), 0.5)

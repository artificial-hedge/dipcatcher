"""Wave-283 robotics-3 module tests."""

import numpy as np

from quant_fund.models.bezier_curve import bezier
from quant_fund.models.fk_dh import fk2
from quant_fund.models.ik_jac import ik2
from quant_fund.models.ray_lidar import cast


def test_fk_zero() -> None:
    np.testing.assert_allclose(fk2(1.0, 1.0, 0.0, 0.0), [2.0, 0.0])


def test_ik_reaches() -> None:
    th, ok = ik2(1.0, 1.0, np.array([1.0, 1.0]))
    assert ok and np.linalg.norm(fk2(1.0, 1.0, th[0], th[1]) - [1.0, 1.0]) < 1e-3


def test_ray_max_empty() -> None:
    assert cast(np.zeros((40, 40), dtype=int), 5.0, 5.0, 0.0) == 20.0


def test_bezier_endpoints() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]])
    np.testing.assert_allclose(bezier(pts, 0.0), pts[0])
    np.testing.assert_allclose(bezier(pts, 1.0), pts[3])

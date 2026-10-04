"""Wave-928 computational-geometry-6 canon tests."""

from __future__ import annotations

from quant_fund.models.beta_skeleton import bench_beta_skeleton
from quant_fund.models.convex_hull_3d import bench_convex_hull_3d
from quant_fund.models.medial_axis import bench_medial_axis
from quant_fund.models.polygon_boolean import bench_polygon_boolean
from quant_fund.models.polygon_centroid import bench_polygon_centroid
from quant_fund.models.shape_context import bench_shape_context


def test_convex_hull_3d():
    assert bench_convex_hull_3d()["synthetic_convex_hull_3d"] == 1.0


def test_polygon_boolean():
    assert bench_polygon_boolean()["synthetic_polygon_boolean"] == 1.0


def test_medial_axis():
    assert bench_medial_axis()["synthetic_medial_axis"] == 1.0


def test_polygon_centroid():
    assert bench_polygon_centroid()["synthetic_polygon_centroid"] == 1.0


def test_shape_context():
    assert bench_shape_context()["synthetic_shape_context"] == 1.0


def test_beta_skeleton():
    assert bench_beta_skeleton()["synthetic_beta_skeleton"] == 1.0

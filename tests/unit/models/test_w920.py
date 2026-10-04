"""Wave-920 computational-geometry-5 canon tests."""

from __future__ import annotations

from quant_fund.models.alpha_shape import bench_alpha_shape
from quant_fund.models.diameter_pair import bench_diameter_pair
from quant_fund.models.min_area_rect import bench_min_area_rect
from quant_fund.models.minkowski_sum_poly import bench_minkowski_sum_poly
from quant_fund.models.monotone_partition import bench_monotone_partition
from quant_fund.models.polygon_triangulate import bench_polygon_triangulate


def test_monotone_partition():
    assert bench_monotone_partition()["synthetic_monotone_partition"] == 1.0


def test_polygon_triangulate():
    assert bench_polygon_triangulate()["synthetic_polygon_triangulate"] == 1.0


def test_min_area_rect():
    assert bench_min_area_rect()["synthetic_min_area_rect"] == 1.0


def test_diameter_pair():
    assert bench_diameter_pair()["synthetic_diameter_pair"] == 1.0


def test_alpha_shape():
    assert bench_alpha_shape()["synthetic_alpha_shape"] == 1.0


def test_minkowski_sum_poly():
    assert bench_minkowski_sum_poly()["synthetic_minkowski_sum_poly"] == 1.0

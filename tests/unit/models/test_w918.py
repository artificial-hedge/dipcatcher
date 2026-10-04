"""Wave-918 data-structures-4/geometry-3 canon tests."""

from __future__ import annotations

from quant_fund.models.fractional_cascade import bench_fractional_cascade
from quant_fund.models.free_list import bench_free_list
from quant_fund.models.halfplane_isect import bench_halfplane_isect
from quant_fund.models.object_pool import bench_object_pool
from quant_fund.models.range_min_query import bench_range_min_query
from quant_fund.models.welzl_circle import bench_welzl_circle


def test_fractional_cascade():
    assert bench_fractional_cascade()["synthetic_fractional_cascade"] == 1.0


def test_range_min_query():
    assert bench_range_min_query()["synthetic_range_min_query"] == 1.0


def test_free_list():
    assert bench_free_list()["synthetic_free_list"] == 1.0


def test_object_pool():
    assert bench_object_pool()["synthetic_object_pool"] == 1.0


def test_welzl_circle():
    assert bench_welzl_circle()["synthetic_welzl_circle"] == 1.0


def test_halfplane_isect():
    assert bench_halfplane_isect()["synthetic_halfplane_isect"] == 1.0

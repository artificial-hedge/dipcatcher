"""Wave-919 computational-geometry-4 canon tests."""

from __future__ import annotations

from quant_fund.models.convex_layers import bench_convex_layers
from quant_fund.models.delaunay_flip import bench_delaunay_flip
from quant_fund.models.polygon_offset import bench_polygon_offset
from quant_fund.models.rotating_sweep import bench_rotating_sweep
from quant_fund.models.visibility_graph import bench_visibility_graph
from quant_fund.models.voronoi_lite import bench_voronoi_lite


def test_voronoi_lite():
    assert bench_voronoi_lite()["synthetic_voronoi_lite"] == 1.0


def test_delaunay_flip():
    assert bench_delaunay_flip()["synthetic_delaunay_flip"] == 1.0


def test_convex_layers():
    assert bench_convex_layers()["synthetic_convex_layers"] == 1.0


def test_polygon_offset():
    assert bench_polygon_offset()["synthetic_polygon_offset"] == 1.0


def test_rotating_sweep():
    assert bench_rotating_sweep()["synthetic_rotating_sweep"] == 1.0


def test_visibility_graph():
    assert bench_visibility_graph()["synthetic_visibility_graph"] == 1.0

"""Wave-912 spatial-index-2 canon tests."""

from __future__ import annotations

from quant_fund.models.hilbert_curve import bench_hilbert_curve
from quant_fund.models.morton_order import bench_morton_order
from quant_fund.models.octree_index import bench_octree_index
from quant_fund.models.range_tree import bench_range_tree
from quant_fund.models.rstar_tree import bench_rstar_tree
from quant_fund.models.z_curve import bench_z_curve


def test_octree_index():
    assert bench_octree_index()["synthetic_octree_index"] == 1.0


def test_range_tree():
    assert bench_range_tree()["synthetic_range_tree"] == 1.0


def test_hilbert_curve():
    assert bench_hilbert_curve()["synthetic_hilbert_curve"] == 1.0


def test_z_curve():
    assert bench_z_curve()["synthetic_z_curve"] == 1.0


def test_morton_order():
    assert bench_morton_order()["synthetic_morton_order"] == 1.0


def test_rstar_tree():
    assert bench_rstar_tree()["synthetic_rstar_tree"] == 1.0

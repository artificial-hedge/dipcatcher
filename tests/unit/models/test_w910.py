"""Wave-910 b-tree family canon tests."""

from __future__ import annotations

from quant_fund.models.b_plus_tree import bench_b_plus_tree
from quant_fund.models.b_star_tree import bench_b_star_tree
from quant_fund.models.b_tree import bench_b_tree
from quant_fund.models.tango_tree import bench_tango_tree
from quant_fund.models.wavl_tree import bench_wavl_tree
from quant_fund.models.weight_balanced_tree import bench_weight_balanced_tree


def test_b_tree():
    assert bench_b_tree()["synthetic_b_tree"] == 1.0


def test_b_plus_tree():
    assert bench_b_plus_tree()["synthetic_b_plus_tree"] == 1.0


def test_b_star_tree():
    assert bench_b_star_tree()["synthetic_b_star_tree"] == 1.0


def test_weight_balanced_tree():
    assert bench_weight_balanced_tree()["synthetic_weight_balanced_tree"] == 1.0


def test_wavl_tree():
    assert bench_wavl_tree()["synthetic_wavl_tree"] == 1.0


def test_tango_tree():
    assert bench_tango_tree()["synthetic_tango_tree"] == 1.0

"""Wave-915 BVP/tree-exotics canon tests."""

from __future__ import annotations

from quant_fund.models.bvp_eigen import bench_bvp_eigen
from quant_fund.models.continuation_bvp import bench_continuation_bvp
from quant_fund.models.fusion_tree import bench_fusion_tree
from quant_fund.models.loser_tree import bench_loser_tree
from quant_fund.models.robbins_bvp import bench_robbins_bvp
from quant_fund.models.superposition_bvp import bench_superposition_bvp


def test_superposition_bvp():
    assert bench_superposition_bvp()["synthetic_superposition_bvp"] == 1.0


def test_continuation_bvp():
    assert bench_continuation_bvp()["synthetic_continuation_bvp"] == 1.0


def test_robbins_bvp():
    assert bench_robbins_bvp()["synthetic_robbins_bvp"] == 1.0


def test_bvp_eigen():
    assert bench_bvp_eigen()["synthetic_bvp_eigen"] == 1.0


def test_loser_tree():
    assert bench_loser_tree()["synthetic_loser_tree"] == 1.0


def test_fusion_tree():
    assert bench_fusion_tree()["synthetic_fusion_tree"] == 1.0

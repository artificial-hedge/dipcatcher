"""Wave-970 subfactor-theory canon tests."""

from __future__ import annotations

from quant_fund.models.fusion_algebra import bench_fusion_algebra
from quant_fund.models.paragroup import bench_paragroup
from quant_fund.models.planar_algebra import bench_planar_algebra
from quant_fund.models.principal_graph import bench_principal_graph
from quant_fund.models.standard_invariant import bench_standard_invariant
from quant_fund.models.subfactor import bench_subfactor


def test_subfactor():
    assert bench_subfactor()["synthetic_subfactor"] == 1.0


def test_standard_invariant():
    assert bench_standard_invariant()["synthetic_standard_invariant"] == 1.0


def test_planar_algebra():
    assert bench_planar_algebra()["synthetic_planar_algebra"] == 1.0


def test_paragroup():
    assert bench_paragroup()["synthetic_paragroup"] == 1.0


def test_principal_graph():
    assert bench_principal_graph()["synthetic_principal_graph"] == 1.0


def test_fusion_algebra():
    assert bench_fusion_algebra()["synthetic_fusion_algebra"] == 1.0

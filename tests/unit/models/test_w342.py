from quant_fund.models.analytic_sets import bench_analytic_sets
from quant_fund.models.arith_hierarchy import bench_arith_hierarchy
from quant_fund.models.borel_hierarchy import bench_borel_hierarchy
from quant_fund.models.forcing_lite import bench_forcing_lite
from quant_fund.models.jump_operator import bench_jump_operator
from quant_fund.models.rice_theorem import bench_rice_theorem


def test_borel_hierarchy():
    assert bench_borel_hierarchy()["synthetic_borel_hierarchy"] == 1.0


def test_analytic_sets():
    assert bench_analytic_sets()["synthetic_analytic_sets"] == 1.0


def test_forcing_lite():
    assert bench_forcing_lite()["synthetic_forcing_lite"] == 1.0


def test_arith_hierarchy():
    assert bench_arith_hierarchy()["synthetic_arith_hierarchy"] == 1.0


def test_jump_operator():
    assert bench_jump_operator()["synthetic_jump_operator"] == 1.0


def test_rice_theorem():
    assert bench_rice_theorem()["synthetic_rice_theorem"] == 1.0

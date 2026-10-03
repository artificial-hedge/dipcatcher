from quant_fund.models.floer_homology import bench_floer_homology
from quant_fund.models.fukaya_cat import bench_fukaya_cat
from quant_fund.models.instanton_floer import bench_instanton_floer
from quant_fund.models.knot_floer import bench_knot_floer
from quant_fund.models.lagrangian_floer import bench_lagrangian_floer
from quant_fund.models.monopole_floer import bench_monopole_floer


def test_floer_homology():
    assert bench_floer_homology()["synthetic_floer_homology"] == 1.0


def test_knot_floer():
    assert bench_knot_floer()["synthetic_knot_floer"] == 1.0


def test_instanton_floer():
    assert bench_instanton_floer()["synthetic_instanton_floer"] == 1.0


def test_monopole_floer():
    assert bench_monopole_floer()["synthetic_monopole_floer"] == 1.0


def test_lagrangian_floer():
    assert bench_lagrangian_floer()["synthetic_lagrangian_floer"] == 1.0


def test_fukaya_cat():
    assert bench_fukaya_cat()["synthetic_fukaya_cat"] == 1.0

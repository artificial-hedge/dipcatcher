from quant_fund.models.cap_product import bench_cap_product
from quant_fund.models.eilenberg_steenrod import bench_eilenberg_steenrod
from quant_fund.models.k_theory import bench_k_theory
from quant_fund.models.obstruction_toy import bench_obstruction_toy
from quant_fund.models.serre_class import bench_serre_class
from quant_fund.models.thom_isom import bench_thom_isom


def test_eilenberg_steenrod():
    assert bench_eilenberg_steenrod()["synthetic_eilenberg_steenrod"] == 1.0


def test_cap_product():
    assert bench_cap_product()["synthetic_cap_product"] == 1.0


def test_thom_isom():
    assert bench_thom_isom()["synthetic_thom_isom"] == 1.0


def test_serre_class():
    assert bench_serre_class()["synthetic_serre_class"] == 1.0


def test_obstruction_toy():
    assert bench_obstruction_toy()["synthetic_obstruction_toy"] == 1.0


def test_k_theory():
    assert bench_k_theory()["synthetic_k_theory"] == 1.0

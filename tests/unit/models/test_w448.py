from quant_fund.models.cohesive_top import bench_cohesive_top
from quant_fund.models.hypercomplete import bench_hypercomplete
from quant_fund.models.infty_topos import bench_infty_topos
from quant_fund.models.object_classif import bench_object_classif
from quant_fund.models.trunc_modal import bench_trunc_modal
from quant_fund.models.univ_colimit import bench_univ_colimit


def test_infty_topos():
    assert bench_infty_topos()["synthetic_infty_topos"] == 1.0


def test_univ_colimit():
    assert bench_univ_colimit()["synthetic_univ_colimit"] == 1.0


def test_object_classif():
    assert bench_object_classif()["synthetic_object_classif"] == 1.0


def test_trunc_modal():
    assert bench_trunc_modal()["synthetic_trunc_modal"] == 1.0


def test_cohesive_top():
    assert bench_cohesive_top()["synthetic_cohesive_top"] == 1.0


def test_hypercomplete():
    assert bench_hypercomplete()["synthetic_hypercomplete"] == 1.0

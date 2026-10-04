from quant_fund.models.brauer_alg import bench_brauer_alg
from quant_fund.models.bz_category import bench_bz_category
from quant_fund.models.casimir_op import bench_casimir_op
from quant_fund.models.hecke_alg import bench_hecke_alg
from quant_fund.models.schur_functor import bench_schur_functor
from quant_fund.models.weight_space import bench_weight_space


def test_schur_functor():
    assert bench_schur_functor()["synthetic_schur_functor"] == 1.0


def test_brauer_alg():
    assert bench_brauer_alg()["synthetic_brauer_alg"] == 1.0


def test_hecke_alg():
    assert bench_hecke_alg()["synthetic_hecke_alg"] == 1.0


def test_casimir_op():
    assert bench_casimir_op()["synthetic_casimir_op"] == 1.0


def test_weight_space():
    assert bench_weight_space()["synthetic_weight_space"] == 1.0


def test_bz_category():
    assert bench_bz_category()["synthetic_bz_category"] == 1.0

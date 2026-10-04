from quant_fund.models.a_infty_alg import bench_a_infty_alg
from quant_fund.models.koszul_duality import (
    bench_koszul_duality,
)
from quant_fund.models.l_infty_alg import bench_l_infty_alg
from quant_fund.models.minimal_model_op import (
    bench_minimal_model_op,
)
from quant_fund.models.operad_cobar import bench_operad_cobar
from quant_fund.models.operadic_bar import bench_operadic_bar


def test_a_infty_alg():
    assert bench_a_infty_alg()["synthetic_a_infty_alg"] == 1.0


def test_l_infty_alg():
    assert bench_l_infty_alg()["synthetic_l_infty_alg"] == 1.0


def test_koszul_duality():
    assert bench_koszul_duality()["synthetic_koszul_duality"] == 1.0


def test_minimal_model_op():
    assert bench_minimal_model_op()["synthetic_minimal_model_op"] == 1.0


def test_operadic_bar():
    assert bench_operadic_bar()["synthetic_operadic_bar"] == 1.0


def test_operad_cobar():
    assert bench_operad_cobar()["synthetic_operad_cobar"] == 1.0

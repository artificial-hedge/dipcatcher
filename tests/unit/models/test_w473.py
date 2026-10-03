from quant_fund.models.cartier_mod import bench_cartier_mod
from quant_fund.models.crystalline_stack import bench_crystalline_stack
from quant_fund.models.cyclotomic2 import bench_cyclotomic2
from quant_fund.models.thh_2 import bench_thh_2
from quant_fund.models.trt_functor import bench_trt_functor
from quant_fund.models.witt_vec2 import bench_witt_vec2


def test_thh_2():
    assert bench_thh_2()["synthetic_thh_2"] == 1.0


def test_cyclotomic2():
    assert bench_cyclotomic2()["synthetic_cyclotomic2"] == 1.0


def test_cartier_mod():
    assert bench_cartier_mod()["synthetic_cartier_mod"] == 1.0


def test_witt_vec2():
    assert bench_witt_vec2()["synthetic_witt_vec2"] == 1.0


def test_crystalline_stack():
    assert bench_crystalline_stack()["synthetic_crystalline_stack"] == 1.0


def test_trt_functor():
    assert bench_trt_functor()["synthetic_trt_functor"] == 1.0

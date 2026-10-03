from quant_fund.models.arthur_param import bench_arthur_param
from quant_fund.models.hecke_alg2 import bench_hecke_alg2
from quant_fund.models.l_function import bench_l_function
from quant_fund.models.satake_param import bench_satake_param
from quant_fund.models.shimura_var import bench_shimura_var
from quant_fund.models.theta_lift import bench_theta_lift


def test_shimura_var():
    assert bench_shimura_var()["synthetic_shimura_var"] == 1.0


def test_l_function():
    assert bench_l_function()["synthetic_l_function"] == 1.0


def test_hecke_alg2():
    assert bench_hecke_alg2()["synthetic_hecke_alg2"] == 1.0


def test_theta_lift():
    assert bench_theta_lift()["synthetic_theta_lift"] == 1.0


def test_arthur_param():
    assert bench_arthur_param()["synthetic_arthur_param"] == 1.0


def test_satake_param():
    assert bench_satake_param()["synthetic_satake_param"] == 1.0

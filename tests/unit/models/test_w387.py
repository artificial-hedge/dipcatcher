from quant_fund.models.concentration_ineq import bench_concentration_ineq
from quant_fund.models.kolmogorov_01 import bench_kolmogorov_01
from quant_fund.models.ldp_theory import bench_ldp_theory
from quant_fund.models.prokhorov_metric import bench_prokhorov_metric
from quant_fund.models.uniform_integrability import bench_uniform_integrability
from quant_fund.models.vitali_conv import bench_vitali_conv


def test_uniform_integrability():
    assert bench_uniform_integrability()["synthetic_uniform_integrability"] == 1.0


def test_vitali_conv():
    assert bench_vitali_conv()["synthetic_vitali_conv"] == 1.0


def test_ldp_theory():
    assert bench_ldp_theory()["synthetic_ldp_theory"] == 1.0


def test_concentration_ineq():
    assert bench_concentration_ineq()["synthetic_concentration_ineq"] == 1.0


def test_kolmogorov_01():
    assert bench_kolmogorov_01()["synthetic_kolmogorov_01"] == 1.0


def test_prokhorov_metric():
    assert bench_prokhorov_metric()["synthetic_prokhorov_metric"] == 1.0

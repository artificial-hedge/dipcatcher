from quant_fund.models.calc_converge import bench_calc_converge
from quant_fund.models.deriv_layer import bench_deriv_layer
from quant_fund.models.excisive_fn import bench_excisive_fn
from quant_fund.models.goodwillie_tower import bench_goodwillie_tower
from quant_fund.models.linearization import bench_linearization
from quant_fund.models.orth_calc import bench_orth_calc


def test_goodwillie_tower():
    assert bench_goodwillie_tower()["synthetic_goodwillie_tower"] == 1.0


def test_excisive_fn():
    assert bench_excisive_fn()["synthetic_excisive_fn"] == 1.0


def test_linearization():
    assert bench_linearization()["synthetic_linearization"] == 1.0


def test_deriv_layer():
    assert bench_deriv_layer()["synthetic_deriv_layer"] == 1.0


def test_calc_converge():
    assert bench_calc_converge()["synthetic_calc_converge"] == 1.0


def test_orth_calc():
    assert bench_orth_calc()["synthetic_orth_calc"] == 1.0

from quant_fund.models.constructible_l import bench_constructible_l
from quant_fund.models.core_model import bench_core_model
from quant_fund.models.large_card import bench_large_card
from quant_fund.models.pcf_theory import bench_pcf_theory
from quant_fund.models.proper_forcing import bench_proper_forcing
from quant_fund.models.square_princ import bench_square_princ


def test_constructible_l():
    assert bench_constructible_l()["synthetic_constructible_l"] == 1.0


def test_large_card():
    assert bench_large_card()["synthetic_large_card"] == 1.0


def test_pcf_theory():
    assert bench_pcf_theory()["synthetic_pcf_theory"] == 1.0


def test_proper_forcing():
    assert bench_proper_forcing()["synthetic_proper_forcing"] == 1.0


def test_core_model():
    assert bench_core_model()["synthetic_core_model"] == 1.0


def test_square_princ():
    assert bench_square_princ()["synthetic_square_princ"] == 1.0

from quant_fund.models.benjamini_ust import bench_benjamini_ust
from quant_fund.models.kirchhoff_matrix import (
    bench_kirchhoff_matrix,
)
from quant_fund.models.lawler_lerw import bench_lawler_lerw
from quant_fund.models.pemantle_ust import bench_pemantle_ust
from quant_fund.models.schramm_lerw import bench_schramm_lerw
from quant_fund.models.wilson_ust import bench_wilson_ust


def test_wilson_ust():
    assert bench_wilson_ust()["synthetic_wilson_ust"] == 1.0


def test_lawler_lerw():
    assert bench_lawler_lerw()["synthetic_lawler_lerw"] == 1.0


def test_benjamini_ust():
    assert bench_benjamini_ust()["synthetic_benjamini_ust"] == 1.0


def test_kirchhoff_matrix():
    assert bench_kirchhoff_matrix()["synthetic_kirchhoff_matrix"] == 1.0


def test_pemantle_ust():
    assert bench_pemantle_ust()["synthetic_pemantle_ust"] == 1.0


def test_schramm_lerw():
    assert bench_schramm_lerw()["synthetic_schramm_lerw"] == 1.0

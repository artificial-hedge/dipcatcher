from quant_fund.models.barbour_stein import bench_barbour_stein
from quant_fund.models.chatt_stein import bench_chatt_stein
from quant_fund.models.chen_stein import bench_chen_stein
from quant_fund.models.ross_stein import bench_ross_stein
from quant_fund.models.stein_equation import (
    bench_stein_equation,
)
from quant_fund.models.stein_method import bench_stein_method


def test_stein_method():
    assert bench_stein_method()["synthetic_stein_method"] == 1.0


def test_stein_equation():
    assert bench_stein_equation()["synthetic_stein_equation"] == 1.0


def test_barbour_stein():
    assert bench_barbour_stein()["synthetic_barbour_stein"] == 1.0


def test_chen_stein():
    assert bench_chen_stein()["synthetic_chen_stein"] == 1.0


def test_ross_stein():
    assert bench_ross_stein()["synthetic_ross_stein"] == 1.0


def test_chatt_stein():
    assert bench_chatt_stein()["synthetic_chatt_stein"] == 1.0

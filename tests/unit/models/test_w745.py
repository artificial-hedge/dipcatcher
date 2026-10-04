from quant_fund.models.bernard_nicola import bench_bernard_nicola
from quant_fund.models.dotsenko_kpz import bench_dotsenko_kpz
from quant_fund.models.hairer_kpz import bench_hairer_kpz
from quant_fund.models.imamura_sasamoto import (
    bench_imamura_sasamoto,
)
from quant_fund.models.spohn_kpz import bench_spohn_kpz
from quant_fund.models.tracy_widom_kpz import bench_tracy_widom_kpz


def test_dotsenko_kpz():
    assert bench_dotsenko_kpz()["synthetic_dotsenko_kpz"] == 1.0


def test_hairer_kpz():
    assert bench_hairer_kpz()["synthetic_hairer_kpz"] == 1.0


def test_bernard_nicola():
    assert bench_bernard_nicola()["synthetic_bernard_nicola"] == 1.0


def test_imamura_sasamoto():
    assert bench_imamura_sasamoto()["synthetic_imamura_sasamoto"] == 1.0


def test_tracy_widom_kpz():
    assert bench_tracy_widom_kpz()["synthetic_tracy_widom_kpz"] == 1.0


def test_spohn_kpz():
    assert bench_spohn_kpz()["synthetic_spohn_kpz"] == 1.0

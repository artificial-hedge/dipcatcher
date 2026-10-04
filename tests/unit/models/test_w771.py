from quant_fund.models.clayton_copula import (
    bench_clayton_copula,
)
from quant_fund.models.copula_gauss import bench_copula_gauss
from quant_fund.models.copula_t import bench_copula_t
from quant_fund.models.frank_copula import bench_frank_copula
from quant_fund.models.gumbel_copula import bench_gumbel_copula
from quant_fund.models.joe_copula import bench_joe_copula


def test_copula_gauss():
    assert bench_copula_gauss()["synthetic_copula_gauss"] == 1.0


def test_copula_t():
    assert bench_copula_t()["synthetic_copula_t"] == 1.0


def test_clayton_copula():
    assert bench_clayton_copula()["synthetic_clayton_copula"] == 1.0


def test_gumbel_copula():
    assert bench_gumbel_copula()["synthetic_gumbel_copula"] == 1.0


def test_frank_copula():
    assert bench_frank_copula()["synthetic_frank_copula"] == 1.0


def test_joe_copula():
    assert bench_joe_copula()["synthetic_joe_copula"] == 1.0

from quant_fund.models.frechet_domain import bench_frechet_domain
from quant_fund.models.gumbel_domain import bench_gumbel_domain
from quant_fund.models.hill_est import bench_hill_est
from quant_fund.models.peak_over import bench_peak_over
from quant_fund.models.pickands_est import bench_pickands_est
from quant_fund.models.weibull_domain import (
    bench_weibull_domain,
)


def test_gumbel_domain():
    assert bench_gumbel_domain()["synthetic_gumbel_domain"] == 1.0


def test_weibull_domain():
    assert bench_weibull_domain()["synthetic_weibull_domain"] == 1.0


def test_frechet_domain():
    assert bench_frechet_domain()["synthetic_frechet_domain"] == 1.0


def test_peak_over():
    assert bench_peak_over()["synthetic_peak_over"] == 1.0


def test_hill_est():
    assert bench_hill_est()["synthetic_hill_est"] == 1.0


def test_pickands_est():
    assert bench_pickands_est()["synthetic_pickands_est"] == 1.0

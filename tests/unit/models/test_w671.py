from quant_fund.models.adams_edge import bench_adams_edge
from quant_fund.models.gray_periodic import bench_gray_periodic
from quant_fund.models.homotopy_exponent import (
    bench_homotopy_exponent,
)
from quant_fund.models.periodic_family import (
    bench_periodic_family,
)
from quant_fund.models.stunted_proj import bench_stunted_proj
from quant_fund.models.unstable_adams2 import (
    bench_unstable_adams2,
)


def test_gray_periodic():
    assert bench_gray_periodic()["synthetic_gray_periodic"] == 1.0


def test_stunted_proj():
    assert bench_stunted_proj()["synthetic_stunted_proj"] == 1.0


def test_adams_edge():
    assert bench_adams_edge()["synthetic_adams_edge"] == 1.0


def test_periodic_family():
    assert bench_periodic_family()["synthetic_periodic_family"] == 1.0


def test_unstable_adams2():
    assert bench_unstable_adams2()["synthetic_unstable_adams2"] == 1.0


def test_homotopy_exponent():
    assert bench_homotopy_exponent()["synthetic_homotopy_exponent"] == 1.0

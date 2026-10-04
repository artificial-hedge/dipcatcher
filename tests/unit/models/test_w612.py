from quant_fund.models.discrete_valuation import (
    bench_discrete_valuation,
)
from quant_fund.models.factorial_ring import (
    bench_factorial_ring,
)
from quant_fund.models.gorenstein_ring import (
    bench_gorenstein_ring,
)
from quant_fund.models.jacobson_ring import (
    bench_jacobson_ring,
)
from quant_fund.models.normal_ring import bench_normal_ring
from quant_fund.models.regular_ring import bench_regular_ring


def test_regular_ring():
    assert bench_regular_ring()["synthetic_regular_ring"] == 1.0


def test_gorenstein_ring():
    assert bench_gorenstein_ring()["synthetic_gorenstein_ring"] == 1.0


def test_normal_ring():
    assert bench_normal_ring()["synthetic_normal_ring"] == 1.0


def test_factorial_ring():
    assert bench_factorial_ring()["synthetic_factorial_ring"] == 1.0


def test_jacobson_ring():
    assert bench_jacobson_ring()["synthetic_jacobson_ring"] == 1.0


def test_discrete_valuation():
    assert bench_discrete_valuation()["synthetic_discrete_valuation"] == 1.0

from quant_fund.models.fractional_heston import (
    bench_fractional_heston,
)
from quant_fund.models.multifactor_rough import (
    bench_multifactor_rough,
)
from quant_fund.models.rough_bergomi import (
    bench_rough_bergomi,
)
from quant_fund.models.rough_sabr import (
    bench_rough_sabr,
)
from quant_fund.models.rough_variance import (
    bench_rough_variance,
)
from quant_fund.models.volterra_sde import (
    bench_volterra_sde,
)


def test_fractional_heston():
    assert bench_fractional_heston()["synthetic_fractional_heston"] == 1.0


def test_rough_bergomi():
    assert bench_rough_bergomi()["synthetic_rough_bergomi"] == 1.0


def test_rough_sabr():
    assert bench_rough_sabr()["synthetic_rough_sabr"] == 1.0


def test_rough_variance():
    assert bench_rough_variance()["synthetic_rough_variance"] == 1.0


def test_volterra_sde():
    assert bench_volterra_sde()["synthetic_volterra_sde"] == 1.0


def test_multifactor_rough():
    assert bench_multifactor_rough()["synthetic_multifactor_rough"] == 1.0

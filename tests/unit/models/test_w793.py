from quant_fund.models.cubature_wiener import (
    bench_cubature_wiener,
)
from quant_fund.models.milstein_scheme import (
    bench_milstein_scheme,
)
from quant_fund.models.rough_vol2 import bench_rough_vol2
from quant_fund.models.stochastic_taylor import (
    bench_stochastic_taylor,
)
from quant_fund.models.wagner_platen import (
    bench_wagner_platen,
)
from quant_fund.models.wong_zakai import (
    bench_wong_zakai,
)


def test_wong_zakai():
    assert bench_wong_zakai()["synthetic_wong_zakai"] == 1.0


def test_stochastic_taylor():
    assert bench_stochastic_taylor()["synthetic_stochastic_taylor"] == 1.0


def test_milstein_scheme():
    assert bench_milstein_scheme()["synthetic_milstein_scheme"] == 1.0


def test_wagner_platen():
    assert bench_wagner_platen()["synthetic_wagner_platen"] == 1.0


def test_cubature_wiener():
    assert bench_cubature_wiener()["synthetic_cubature_wiener"] == 1.0


def test_rough_vol2():
    assert bench_rough_vol2()["synthetic_rough_vol2"] == 1.0

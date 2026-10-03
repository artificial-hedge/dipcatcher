from quant_fund.models.absolute_cont import (
    bench_absolute_cont,
)
from quant_fund.models.density_bound import (
    bench_density_bound,
)
from quant_fund.models.malliavin_cov import (
    bench_malliavin_cov,
)
from quant_fund.models.nualart_zakai import (
    bench_nualart_zakai,
)
from quant_fund.models.smoothness_h import (
    bench_smoothness_h,
)
from quant_fund.models.watanabe_map import (
    bench_watanabe_map,
)


def test_nualart_zakai():
    assert bench_nualart_zakai()["synthetic_nualart_zakai"] == 1.0


def test_watanabe_map():
    assert bench_watanabe_map()["synthetic_watanabe_map"] == 1.0


def test_malliavin_cov():
    assert bench_malliavin_cov()["synthetic_malliavin_cov"] == 1.0


def test_density_bound():
    assert bench_density_bound()["synthetic_density_bound"] == 1.0


def test_absolute_cont():
    assert bench_absolute_cont()["synthetic_absolute_cont"] == 1.0


def test_smoothness_h():
    assert bench_smoothness_h()["synthetic_smoothness_h"] == 1.0

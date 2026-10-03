from quant_fund.models.continuous_map import (
    bench_continuous_map,
)
from quant_fund.models.delta_method import (
    bench_delta_method,
)
from quant_fund.models.empirical_bridge import (
    bench_empirical_bridge,
)
from quant_fund.models.kmt_approx import (
    bench_kmt_approx,
)
from quant_fund.models.porte_manteau import (
    bench_porte_manteau,
)
from quant_fund.models.skorohod_embed import (
    bench_skorohod_embed,
)


def test_porte_manteau():
    assert bench_porte_manteau()["synthetic_porte_manteau"] == 1.0


def test_continuous_map():
    assert bench_continuous_map()["synthetic_continuous_map"] == 1.0


def test_delta_method():
    assert bench_delta_method()["synthetic_delta_method"] == 1.0


def test_skorohod_embed():
    assert bench_skorohod_embed()["synthetic_skorohod_embed"] == 1.0


def test_kmt_approx():
    assert bench_kmt_approx()["synthetic_kmt_approx"] == 1.0


def test_empirical_bridge():
    assert bench_empirical_bridge()["synthetic_empirical_bridge"] == 1.0

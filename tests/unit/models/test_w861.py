from quant_fund.models.ars_imex import (
    bench_ars_imex,
)
from quant_fund.models.dirk_scheme import (
    bench_dirk_scheme,
)
from quant_fund.models.exponential_euler import (
    bench_exponential_euler,
)
from quant_fund.models.imex_rk import (
    bench_imex_rk,
)
from quant_fund.models.rosenbrock_w import (
    bench_rosenbrock_w,
)
from quant_fund.models.ssp_rk import (
    bench_ssp_rk,
)


def test_imex_rk():
    assert bench_imex_rk()["synthetic_imex_rk"] == 1.0


def test_ssp_rk():
    assert bench_ssp_rk()["synthetic_ssp_rk"] == 1.0


def test_exponential_euler():
    assert bench_exponential_euler()["synthetic_exponential_euler"] == 1.0


def test_rosenbrock_w():
    assert bench_rosenbrock_w()["synthetic_rosenbrock_w"] == 1.0


def test_ars_imex():
    assert bench_ars_imex()["synthetic_ars_imex"] == 1.0


def test_dirk_scheme():
    assert bench_dirk_scheme()["synthetic_dirk_scheme"] == 1.0

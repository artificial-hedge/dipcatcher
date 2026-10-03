from quant_fund.models.aitken_steffensen import (
    bench_aitken_steffensen,
)
from quant_fund.models.bulirsch_stoer import (
    bench_bulirsch_stoer,
)
from quant_fund.models.muller_root import (
    bench_muller_root,
)
from quant_fund.models.regula_falsi import (
    bench_regula_falsi,
)
from quant_fund.models.richardson_limit import (
    bench_richardson_limit,
)
from quant_fund.models.secant_root import (
    bench_secant_root,
)


def test_secant_root():
    assert bench_secant_root()["synthetic_secant_root"] == 1.0


def test_regula_falsi():
    assert bench_regula_falsi()["synthetic_regula_falsi"] == 1.0


def test_muller_root():
    assert bench_muller_root()["synthetic_muller_root"] == 1.0


def test_aitken_steffensen():
    assert bench_aitken_steffensen()["synthetic_aitken_steffensen"] == 1.0


def test_richardson_limit():
    assert bench_richardson_limit()["synthetic_richardson_limit"] == 1.0


def test_bulirsch_stoer():
    assert bench_bulirsch_stoer()["synthetic_bulirsch_stoer"] == 1.0

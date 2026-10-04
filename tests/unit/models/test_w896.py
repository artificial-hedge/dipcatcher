from quant_fund.models.backward_euler import (
    bench_backward_euler,
)
from quant_fund.models.bogacki_shampine import (
    bench_bogacki_shampine,
)
from quant_fund.models.cash_karp import (
    bench_cash_karp,
)
from quant_fund.models.dormand_prince import (
    bench_dormand_prince,
)
from quant_fund.models.fehlberg_rk import (
    bench_fehlberg_rk,
)
from quant_fund.models.predictor_corrector import (
    bench_predictor_corrector,
)


def test_fehlberg_rk():
    assert bench_fehlberg_rk()["synthetic_fehlberg_rk"] == 1.0


def test_dormand_prince():
    assert bench_dormand_prince()["synthetic_dormand_prince"] == 1.0


def test_cash_karp():
    assert bench_cash_karp()["synthetic_cash_karp"] == 1.0


def test_bogacki_shampine():
    assert bench_bogacki_shampine()["synthetic_bogacki_shampine"] == 1.0


def test_backward_euler():
    assert bench_backward_euler()["synthetic_backward_euler"] == 1.0


def test_predictor_corrector():
    assert bench_predictor_corrector()["synthetic_predictor_corrector"] == 1.0

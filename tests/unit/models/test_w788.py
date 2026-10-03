from quant_fund.models.clark_ocone import (
    bench_clark_ocone,
)
from quant_fund.models.divergence_op import (
    bench_divergence_op,
)
from quant_fund.models.nourdin_peccati import (
    bench_nourdin_peccati,
)
from quant_fund.models.nualart_pardoux import (
    bench_nualart_pardoux,
)
from quant_fund.models.skorohod_int import (
    bench_skorohod_int,
)
from quant_fund.models.wiener_chaos import (
    bench_wiener_chaos,
)


def test_clark_ocone():
    assert bench_clark_ocone()["synthetic_clark_ocone"] == 1.0


def test_nualart_pardoux():
    assert bench_nualart_pardoux()["synthetic_nualart_pardoux"] == 1.0


def test_divergence_op():
    assert bench_divergence_op()["synthetic_divergence_op"] == 1.0


def test_wiener_chaos():
    assert bench_wiener_chaos()["synthetic_wiener_chaos"] == 1.0


def test_skorohod_int():
    assert bench_skorohod_int()["synthetic_skorohod_int"] == 1.0


def test_nourdin_peccati():
    assert bench_nourdin_peccati()["synthetic_nourdin_peccati"] == 1.0

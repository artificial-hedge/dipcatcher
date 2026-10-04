from quant_fund.models.cambrian_pp import (
    bench_cambrian_pp,
)
from quant_fund.models.ergodic_pp import (
    bench_ergodic_pp,
)
from quant_fund.models.gneding_metric import (
    bench_gneding_metric,
)
from quant_fund.models.j_function import (
    bench_j_function,
)
from quant_fund.models.papangelou import (
    bench_papangelou,
)
from quant_fund.models.void_prob import (
    bench_void_prob,
)


def test_cambrian_pp():
    assert bench_cambrian_pp()["synthetic_cambrian_pp"] == 1.0


def test_papangelou():
    assert bench_papangelou()["synthetic_papangelou"] == 1.0


def test_gneding_metric():
    assert bench_gneding_metric()["synthetic_gneding_metric"] == 1.0


def test_void_prob():
    assert bench_void_prob()["synthetic_void_prob"] == 1.0


def test_j_function():
    assert bench_j_function()["synthetic_j_function"] == 1.0


def test_ergodic_pp():
    assert bench_ergodic_pp()["synthetic_ergodic_pp"] == 1.0

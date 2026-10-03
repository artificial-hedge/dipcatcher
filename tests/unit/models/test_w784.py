from quant_fund.models.girsanov_thm2 import (
    bench_girsanov_thm2,
)
from quant_fund.models.levy_khinchine import (
    bench_levy_khinchine,
)
from quant_fund.models.levy_measure import (
    bench_levy_measure,
)
from quant_fund.models.self_decomp import (
    bench_self_decomp,
)
from quant_fund.models.stable_levy import (
    bench_stable_levy,
)
from quant_fund.models.subordinator import (
    bench_subordinator,
)


def test_levy_khinchine():
    assert bench_levy_khinchine()["synthetic_levy_khinchine"] == 1.0


def test_subordinator():
    assert bench_subordinator()["synthetic_subordinator"] == 1.0


def test_stable_levy():
    assert bench_stable_levy()["synthetic_stable_levy"] == 1.0


def test_self_decomp():
    assert bench_self_decomp()["synthetic_self_decomp"] == 1.0


def test_levy_measure():
    assert bench_levy_measure()["synthetic_levy_measure"] == 1.0


def test_girsanov_thm2():
    assert bench_girsanov_thm2()["synthetic_girsanov_thm2"] == 1.0

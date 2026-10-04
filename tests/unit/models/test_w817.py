from quant_fund.models.ladder_height import (
    bench_ladder_height,
)
from quant_fund.models.levy_fluct import (
    bench_levy_fluct,
)
from quant_fund.models.overshoot_levy import (
    bench_overshoot_levy,
)
from quant_fund.models.renewal_measure import (
    bench_renewal_measure,
)
from quant_fund.models.spitzer_levy import (
    bench_spitzer_levy,
)
from quant_fund.models.wiener_hopf_f import (
    bench_wiener_hopf_f,
)


def test_wiener_hopf_f():
    assert bench_wiener_hopf_f()["synthetic_wiener_hopf_f"] == 1.0


def test_ladder_height():
    assert bench_ladder_height()["synthetic_ladder_height"] == 1.0


def test_renewal_measure():
    assert bench_renewal_measure()["synthetic_renewal_measure"] == 1.0


def test_overshoot_levy():
    assert bench_overshoot_levy()["synthetic_overshoot_levy"] == 1.0


def test_levy_fluct():
    assert bench_levy_fluct()["synthetic_levy_fluct"] == 1.0


def test_spitzer_levy():
    assert bench_spitzer_levy()["synthetic_spitzer_levy"] == 1.0

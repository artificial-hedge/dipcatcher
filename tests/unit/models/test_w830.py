from quant_fund.models.azuma_ineq import (
    bench_azuma_ineq,
)
from quant_fund.models.bounded_diff import (
    bench_bounded_diff,
)
from quant_fund.models.efron_stein import (
    bench_efron_stein,
)
from quant_fund.models.hoeffding_ineq import (
    bench_hoeffding_ineq,
)
from quant_fund.models.mcdiarmid_ineq import (
    bench_mcdiarmid_ineq,
)
from quant_fund.models.talagrand_ineq import (
    bench_talagrand_ineq,
)


def test_azuma_ineq():
    assert bench_azuma_ineq()["synthetic_azuma_ineq"] == 1.0


def test_mcdiarmid_ineq():
    assert bench_mcdiarmid_ineq()["synthetic_mcdiarmid_ineq"] == 1.0


def test_talagrand_ineq():
    assert bench_talagrand_ineq()["synthetic_talagrand_ineq"] == 1.0


def test_efron_stein():
    assert bench_efron_stein()["synthetic_efron_stein"] == 1.0


def test_bounded_diff():
    assert bench_bounded_diff()["synthetic_bounded_diff"] == 1.0


def test_hoeffding_ineq():
    assert bench_hoeffding_ineq()["synthetic_hoeffding_ineq"] == 1.0

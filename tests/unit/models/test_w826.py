from quant_fund.models.bj_ineq import (
    bench_bj_ineq,
)
from quant_fund.models.doob_ineq import (
    bench_doob_ineq,
)
from quant_fund.models.etemadi_ineq import (
    bench_etemadi_ineq,
)
from quant_fund.models.kolmogorov_ineq import (
    bench_kolmogorov_ineq,
)
from quant_fund.models.levy_ineq import (
    bench_levy_ineq,
)
from quant_fund.models.max_ineq import (
    bench_max_ineq,
)


def test_doob_ineq():
    assert bench_doob_ineq()["synthetic_doob_ineq"] == 1.0


def test_max_ineq():
    assert bench_max_ineq()["synthetic_max_ineq"] == 1.0


def test_bj_ineq():
    assert bench_bj_ineq()["synthetic_bj_ineq"] == 1.0


def test_kolmogorov_ineq():
    assert bench_kolmogorov_ineq()["synthetic_kolmogorov_ineq"] == 1.0


def test_etemadi_ineq():
    assert bench_etemadi_ineq()["synthetic_etemadi_ineq"] == 1.0


def test_levy_ineq():
    assert bench_levy_ineq()["synthetic_levy_ineq"] == 1.0

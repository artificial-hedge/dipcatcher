from quant_fund.models.asymptotic_series import (
    bench_asymptotic_series,
)
from quant_fund.models.borel_resum import (
    bench_borel_resum,
)
from quant_fund.models.poincare_expansion import (
    bench_poincare_expansion,
)
from quant_fund.models.stationary_phase import (
    bench_stationary_phase,
)
from quant_fund.models.steepest_descent import (
    bench_steepest_descent,
)
from quant_fund.models.wkb_approx import (
    bench_wkb_approx,
)


def test_asymptotic_series():
    assert bench_asymptotic_series()["synthetic_asymptotic_series"] == 1.0


def test_poincare_expansion():
    assert bench_poincare_expansion()["synthetic_poincare_expansion"] == 1.0


def test_steepest_descent():
    assert bench_steepest_descent()["synthetic_steepest_descent"] == 1.0


def test_stationary_phase():
    assert bench_stationary_phase()["synthetic_stationary_phase"] == 1.0


def test_borel_resum():
    assert bench_borel_resum()["synthetic_borel_resum"] == 1.0


def test_wkb_approx():
    assert bench_wkb_approx()["synthetic_wkb_approx"] == 1.0

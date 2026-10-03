from quant_fund.models.cv_optimal import (
    bench_cv_optimal,
)
from quant_fund.models.is_drift import (
    bench_is_drift,
)
from quant_fund.models.min_var_closure import (
    bench_min_var_closure,
)
from quant_fund.models.nest_accel import (
    bench_nest_accel,
)
from quant_fund.models.subgradient_descent import (
    bench_subgradient_descent,
)
from quant_fund.models.tangent_predictor import (
    bench_tangent_predictor,
)


def test_tangent_predictor():
    assert bench_tangent_predictor()["synthetic_tangent_predictor"] == 1.0


def test_is_drift():
    assert bench_is_drift()["synthetic_is_drift"] == 1.0


def test_cv_optimal():
    assert bench_cv_optimal()["synthetic_cv_optimal"] == 1.0


def test_nest_accel():
    assert bench_nest_accel()["synthetic_nest_accel"] == 1.0


def test_subgradient_descent():
    assert bench_subgradient_descent()["synthetic_subgradient_descent"] == 1.0


def test_min_var_closure():
    assert bench_min_var_closure()["synthetic_min_var_closure"] == 1.0

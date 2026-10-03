from quant_fund.models.dual_weighted_res import (
    bench_dual_weighted_res,
)
from quant_fund.models.equilibrated_flux import (
    bench_equilibrated_flux,
)
from quant_fund.models.goal_oriented import (
    bench_goal_oriented,
)
from quant_fund.models.recovery_error import (
    bench_recovery_error,
)
from quant_fund.models.residual_estimator import (
    bench_residual_estimator,
)
from quant_fund.models.zienkiewicz_zhu import (
    bench_zienkiewicz_zhu,
)


def test_residual_estimator():
    assert bench_residual_estimator()["synthetic_residual_estimator"] == 1.0


def test_zienkiewicz_zhu():
    assert bench_zienkiewicz_zhu()["synthetic_zienkiewicz_zhu"] == 1.0


def test_recovery_error():
    assert bench_recovery_error()["synthetic_recovery_error"] == 1.0


def test_dual_weighted_res():
    assert bench_dual_weighted_res()["synthetic_dual_weighted_res"] == 1.0


def test_goal_oriented():
    assert bench_goal_oriented()["synthetic_goal_oriented"] == 1.0


def test_equilibrated_flux():
    assert bench_equilibrated_flux()["synthetic_equilibrated_flux"] == 1.0

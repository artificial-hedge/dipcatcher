from quant_fund.models.coarsening_mark import (
    bench_coarsening_mark,
)
from quant_fund.models.covello_est import (
    bench_covello_est,
)
from quant_fund.models.dual_goal_est import (
    bench_dual_goal_est,
)
from quant_fund.models.galerkin_least_sq import (
    bench_galerkin_least_sq,
)
from quant_fund.models.pseudospectral_coll import (
    bench_pseudospectral_coll,
)
from quant_fund.models.tau_method import (
    bench_tau_method,
)


def test_covello_est():
    assert bench_covello_est()["synthetic_covello_est"] == 1.0


def test_dual_goal_est():
    assert bench_dual_goal_est()["synthetic_dual_goal_est"] == 1.0


def test_pseudospectral_coll():
    assert bench_pseudospectral_coll()["synthetic_pseudospectral_coll"] == 1.0


def test_tau_method():
    assert bench_tau_method()["synthetic_tau_method"] == 1.0


def test_galerkin_least_sq():
    assert bench_galerkin_least_sq()["synthetic_galerkin_least_sq"] == 1.0


def test_coarsening_mark():
    assert bench_coarsening_mark()["synthetic_coarsening_mark"] == 1.0

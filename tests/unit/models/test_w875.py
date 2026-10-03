from quant_fund.models.adaptive_finite import (
    bench_adaptive_finite,
)
from quant_fund.models.adaptive_marking import (
    bench_adaptive_marking,
)
from quant_fund.models.convergence_theory import (
    bench_convergence_theory,
)
from quant_fund.models.dorfler_marking import (
    bench_dorfler_marking,
)
from quant_fund.models.goal_adaptive import (
    bench_goal_adaptive,
)
from quant_fund.models.hierarchical_est import (
    bench_hierarchical_est,
)


def test_adaptive_marking():
    assert bench_adaptive_marking()["synthetic_adaptive_marking"] == 1.0


def test_hierarchical_est():
    assert bench_hierarchical_est()["synthetic_hierarchical_est"] == 1.0


def test_dorfler_marking():
    assert bench_dorfler_marking()["synthetic_dorfler_marking"] == 1.0


def test_convergence_theory():
    assert bench_convergence_theory()["synthetic_convergence_theory"] == 1.0


def test_adaptive_finite():
    assert bench_adaptive_finite()["synthetic_adaptive_finite"] == 1.0


def test_goal_adaptive():
    assert bench_goal_adaptive()["synthetic_goal_adaptive"] == 1.0

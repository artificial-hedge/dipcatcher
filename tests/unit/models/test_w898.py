from quant_fund.models.collocation_bvp import (
    bench_collocation_bvp,
)
from quant_fund.models.finite_diff_bvp import (
    bench_finite_diff_bvp,
)
from quant_fund.models.multiple_shooting import (
    bench_multiple_shooting,
)
from quant_fund.models.relaxation_bvp import (
    bench_relaxation_bvp,
)
from quant_fund.models.riccati_bvp import (
    bench_riccati_bvp,
)
from quant_fund.models.shooting_bvp import (
    bench_shooting_bvp,
)


def test_shooting_bvp():
    assert bench_shooting_bvp()["synthetic_shooting_bvp"] == 1.0


def test_multiple_shooting():
    assert bench_multiple_shooting()["synthetic_multiple_shooting"] == 1.0


def test_collocation_bvp():
    assert bench_collocation_bvp()["synthetic_collocation_bvp"] == 1.0


def test_finite_diff_bvp():
    assert bench_finite_diff_bvp()["synthetic_finite_diff_bvp"] == 1.0


def test_relaxation_bvp():
    assert bench_relaxation_bvp()["synthetic_relaxation_bvp"] == 1.0


def test_riccati_bvp():
    assert bench_riccati_bvp()["synthetic_riccati_bvp"] == 1.0

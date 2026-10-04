from quant_fund.models.dg_discretization import (
    bench_dg_discretization,
)
from quant_fund.models.limiter_tvb import (
    bench_limiter_tvb,
)
from quant_fund.models.modal_basis import (
    bench_modal_basis,
)
from quant_fund.models.numerical_flux_dg import (
    bench_numerical_flux_dg,
)
from quant_fund.models.penalty_dg import (
    bench_penalty_dg,
)
from quant_fund.models.rkdg_step import (
    bench_rkdg_step,
)


def test_dg_discretization():
    assert bench_dg_discretization()["synthetic_dg_discretization"] == 1.0


def test_numerical_flux_dg():
    assert bench_numerical_flux_dg()["synthetic_numerical_flux_dg"] == 1.0


def test_penalty_dg():
    assert bench_penalty_dg()["synthetic_penalty_dg"] == 1.0


def test_modal_basis():
    assert bench_modal_basis()["synthetic_modal_basis"] == 1.0


def test_limiter_tvb():
    assert bench_limiter_tvb()["synthetic_limiter_tvb"] == 1.0


def test_rkdg_step():
    assert bench_rkdg_step()["synthetic_rkdg_step"] == 1.0

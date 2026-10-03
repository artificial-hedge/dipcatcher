from quant_fund.models.intrusive_pce import (
    bench_intrusive_pce,
)
from quant_fund.models.nonintrusive_pce import (
    bench_nonintrusive_pce,
)
from quant_fund.models.poly_chaos_uq import (
    bench_poly_chaos_uq,
)
from quant_fund.models.stochastic_colloc import (
    bench_stochastic_colloc,
)
from quant_fund.models.stochastic_fem import (
    bench_stochastic_fem,
)
from quant_fund.models.stochastic_galerkin import (
    bench_stochastic_galerkin,
)


def test_stochastic_galerkin():
    assert bench_stochastic_galerkin()["synthetic_stochastic_galerkin"] == 1.0


def test_poly_chaos_uq():
    assert bench_poly_chaos_uq()["synthetic_poly_chaos_uq"] == 1.0


def test_intrusive_pce():
    assert bench_intrusive_pce()["synthetic_intrusive_pce"] == 1.0


def test_nonintrusive_pce():
    assert bench_nonintrusive_pce()["synthetic_nonintrusive_pce"] == 1.0


def test_stochastic_colloc():
    assert bench_stochastic_colloc()["synthetic_stochastic_colloc"] == 1.0


def test_stochastic_fem():
    assert bench_stochastic_fem()["synthetic_stochastic_fem"] == 1.0

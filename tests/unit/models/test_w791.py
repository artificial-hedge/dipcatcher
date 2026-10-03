from quant_fund.models.doering_mueller import (
    bench_doering_mueller,
)
from quant_fund.models.kpz_equation import (
    bench_kpz_equation,
)
from quant_fund.models.paracontrolled_spde import (
    bench_paracontrolled_spde,
)
from quant_fund.models.quasilinear_spde import (
    bench_quasilinear_spde,
)
from quant_fund.models.spde_heat import bench_spde_heat
from quant_fund.models.stochastic_burgers import (
    bench_stochastic_burgers,
)


def test_spde_heat():
    assert bench_spde_heat()["synthetic_spde_heat"] == 1.0


def test_stochastic_burgers():
    assert bench_stochastic_burgers()["synthetic_stochastic_burgers"] == 1.0


def test_kpz_equation():
    assert bench_kpz_equation()["synthetic_kpz_equation"] == 1.0


def test_doering_mueller():
    assert bench_doering_mueller()["synthetic_doering_mueller"] == 1.0


def test_quasilinear_spde():
    assert bench_quasilinear_spde()["synthetic_quasilinear_spde"] == 1.0


def test_paracontrolled_spde():
    assert bench_paracontrolled_spde()["synthetic_paracontrolled_spde"] == 1.0

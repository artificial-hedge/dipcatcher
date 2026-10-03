from quant_fund.models.bfgs_update import (
    bench_bfgs_update,
)
from quant_fund.models.iga_colloc import (
    bench_iga_colloc,
)
from quant_fund.models.lebesgue_const import (
    bench_lebesgue_const,
)
from quant_fund.models.newton_armijo import (
    bench_newton_armijo,
)
from quant_fund.models.trimmed_cad import (
    bench_trimmed_cad,
)
from quant_fund.models.trust_region_dogleg import (
    bench_trust_region_dogleg,
)


def test_trust_region_dogleg():
    assert bench_trust_region_dogleg()["synthetic_trust_region_dogleg"] == 1.0


def test_bfgs_update():
    assert bench_bfgs_update()["synthetic_bfgs_update"] == 1.0


def test_lebesgue_const():
    assert bench_lebesgue_const()["synthetic_lebesgue_const"] == 1.0


def test_iga_colloc():
    assert bench_iga_colloc()["synthetic_iga_colloc"] == 1.0


def test_trimmed_cad():
    assert bench_trimmed_cad()["synthetic_trimmed_cad"] == 1.0


def test_newton_armijo():
    assert bench_newton_armijo()["synthetic_newton_armijo"] == 1.0

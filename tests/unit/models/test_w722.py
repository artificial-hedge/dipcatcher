from quant_fund.models.borel_motivic import bench_borel_motivic
from quant_fund.models.deligne_period import bench_deligne_period
from quant_fund.models.motivic_multiple_zeta import (
    bench_motivic_multiple_zeta,
)
from quant_fund.models.period_poly import bench_period_poly
from quant_fund.models.specialization_motive import (
    bench_specialization_motive,
)
from quant_fund.models.zagier_polylog import bench_zagier_polylog


def test_period_poly():
    assert bench_period_poly()["synthetic_period_poly"] == 1.0


def test_specialization_motive():
    assert bench_specialization_motive()["synthetic_specialization_motive"] == 1.0


def test_borel_motivic():
    assert bench_borel_motivic()["synthetic_borel_motivic"] == 1.0


def test_zagier_polylog():
    assert bench_zagier_polylog()["synthetic_zagier_polylog"] == 1.0


def test_deligne_period():
    assert bench_deligne_period()["synthetic_deligne_period"] == 1.0


def test_motivic_multiple_zeta():
    assert bench_motivic_multiple_zeta()["synthetic_motivic_multiple_zeta"] == 1.0

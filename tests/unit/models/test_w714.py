from quant_fund.models.derived_proper2 import (
    bench_derived_proper2,
)
from quant_fund.models.derived_separated2 import (
    bench_derived_separated2,
)
from quant_fund.models.motivic_functor import (
    bench_motivic_functor,
)
from quant_fund.models.motivic_nerve import bench_motivic_nerve
from quant_fund.models.motivic_partial import (
    bench_motivic_partial,
)
from quant_fund.models.motivic_total import bench_motivic_total


def test_motivic_total():
    assert bench_motivic_total()["synthetic_motivic_total"] == 1.0


def test_motivic_partial():
    assert bench_motivic_partial()["synthetic_motivic_partial"] == 1.0


def test_motivic_functor():
    assert bench_motivic_functor()["synthetic_motivic_functor"] == 1.0


def test_motivic_nerve():
    assert bench_motivic_nerve()["synthetic_motivic_nerve"] == 1.0


def test_derived_proper2():
    assert bench_derived_proper2()["synthetic_derived_proper2"] == 1.0


def test_derived_separated2():
    assert bench_derived_separated2()["synthetic_derived_separated2"] == 1.0

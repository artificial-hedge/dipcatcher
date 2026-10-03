from quant_fund.models.derived_loop import bench_derived_loop
from quant_fund.models.derived_tangent import (
    bench_derived_tangent,
)
from quant_fund.models.dg_algebra import bench_dg_algebra
from quant_fund.models.e_infinity_ring import (
    bench_e_infinity_ring,
)
from quant_fund.models.structured_space import (
    bench_structured_space,
)
from quant_fund.models.virtual_fund import bench_virtual_fund


def test_dg_algebra():
    assert bench_dg_algebra()["synthetic_dg_algebra"] == 1.0


def test_derived_loop():
    assert bench_derived_loop()["synthetic_derived_loop"] == 1.0


def test_derived_tangent():
    assert bench_derived_tangent()["synthetic_derived_tangent"] == 1.0


def test_virtual_fund():
    assert bench_virtual_fund()["synthetic_virtual_fund"] == 1.0


def test_structured_space():
    assert bench_structured_space()["synthetic_structured_space"] == 1.0


def test_e_infinity_ring():
    assert bench_e_infinity_ring()["synthetic_e_infinity_ring"] == 1.0

from quant_fund.models.cartesian_closed import (
    bench_cartesian_closed,
)
from quant_fund.models.coherent_topos import bench_coherent_topos
from quant_fund.models.internal_logic import (
    bench_internal_logic,
)
from quant_fund.models.power_object import bench_power_object
from quant_fund.models.pretopos import bench_pretopos
from quant_fund.models.subobject_lattice import (
    bench_subobject_lattice,
)


def test_cartesian_closed():
    assert bench_cartesian_closed()["synthetic_cartesian_closed"] == 1.0


def test_internal_logic():
    assert bench_internal_logic()["synthetic_internal_logic"] == 1.0


def test_subobject_lattice():
    assert bench_subobject_lattice()["synthetic_subobject_lattice"] == 1.0


def test_power_object():
    assert bench_power_object()["synthetic_power_object"] == 1.0


def test_pretopos():
    assert bench_pretopos()["synthetic_pretopos"] == 1.0


def test_coherent_topos():
    assert bench_coherent_topos()["synthetic_coherent_topos"] == 1.0

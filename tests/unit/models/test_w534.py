from quant_fund.models.balayage import bench_balayage
from quant_fund.models.capacity_theory import bench_capacity_theory
from quant_fund.models.fine_topology import bench_fine_topology
from quant_fund.models.green_fn import bench_green_fn
from quant_fund.models.harmonic_fn import bench_harmonic_fn
from quant_fund.models.potential_thy import bench_potential_thy


def test_harmonic_fn():
    assert bench_harmonic_fn()["synthetic_harmonic_fn"] == 1.0


def test_potential_thy():
    assert bench_potential_thy()["synthetic_potential_thy"] == 1.0


def test_capacity_theory():
    assert bench_capacity_theory()["synthetic_capacity_theory"] == 1.0


def test_balayage():
    assert bench_balayage()["synthetic_balayage"] == 1.0


def test_green_fn():
    assert bench_green_fn()["synthetic_green_fn"] == 1.0


def test_fine_topology():
    assert bench_fine_topology()["synthetic_fine_topology"] == 1.0

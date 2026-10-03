"""Wave-982 potential-theory-2 canon tests."""

from __future__ import annotations

from quant_fund.models.boundary_regular import bench_boundary_regular
from quant_fund.models.capacitary_pot import bench_capacitary_pot
from quant_fund.models.dirichlet_problem import bench_dirichlet_problem
from quant_fund.models.energy_principle import bench_energy_principle
from quant_fund.models.equilibrium_measure import bench_equilibrium_measure
from quant_fund.models.thin_set import bench_thin_set


def test_dirichlet_problem():
    assert bench_dirichlet_problem()["synthetic_dirichlet_problem"] == 1.0


def test_energy_principle():
    assert bench_energy_principle()["synthetic_energy_principle"] == 1.0


def test_equilibrium_measure():
    assert bench_equilibrium_measure()["synthetic_equilibrium_measure"] == 1.0


def test_thin_set():
    assert bench_thin_set()["synthetic_thin_set"] == 1.0


def test_boundary_regular():
    assert bench_boundary_regular()["synthetic_boundary_regular"] == 1.0


def test_capacitary_pot():
    assert bench_capacitary_pot()["synthetic_capacitary_pot"] == 1.0

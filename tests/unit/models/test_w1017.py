"""Wave-1017 continuum-mechanics canon tests."""

from __future__ import annotations

from quant_fund.models.navier_cauchy import bench_navier_cauchy
from quant_fund.models.plasticity import bench_plasticity
from quant_fund.models.poroelasticity import bench_poroelasticity
from quant_fund.models.rheology import bench_rheology
from quant_fund.models.stress_tensor import bench_stress_tensor
from quant_fund.models.viscoelasticity import bench_viscoelasticity


def test_navier_cauchy():
    assert bench_navier_cauchy()["synthetic_navier_cauchy"] == 1.0


def test_stress_tensor():
    assert bench_stress_tensor()["synthetic_stress_tensor"] == 1.0


def test_rheology():
    assert bench_rheology()["synthetic_rheology"] == 1.0


def test_viscoelasticity():
    assert bench_viscoelasticity()["synthetic_viscoelasticity"] == 1.0


def test_plasticity():
    assert bench_plasticity()["synthetic_plasticity"] == 1.0


def test_poroelasticity():
    assert bench_poroelasticity()["synthetic_poroelasticity"] == 1.0

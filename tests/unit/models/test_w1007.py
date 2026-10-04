"""Wave-1007 statistical-mechanics canon tests."""

from __future__ import annotations

from quant_fund.models.bose_einstein import bench_bose_einstein
from quant_fund.models.fermi_dirac import bench_fermi_dirac
from quant_fund.models.free_energy import bench_free_energy
from quant_fund.models.gibbs_measure import bench_gibbs_measure
from quant_fund.models.ising_model import bench_ising_model
from quant_fund.models.partition_function import bench_partition_function


def test_ising_model():
    assert bench_ising_model()["synthetic_ising_model"] == 1.0


def test_partition_function():
    assert bench_partition_function()["synthetic_partition_function"] == 1.0


def test_bose_einstein():
    assert bench_bose_einstein()["synthetic_bose_einstein"] == 1.0


def test_fermi_dirac():
    assert bench_fermi_dirac()["synthetic_fermi_dirac"] == 1.0


def test_gibbs_measure():
    assert bench_gibbs_measure()["synthetic_gibbs_measure"] == 1.0


def test_free_energy():
    assert bench_free_energy()["synthetic_free_energy"] == 1.0

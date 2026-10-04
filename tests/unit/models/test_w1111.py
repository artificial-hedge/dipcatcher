"""Wave-1111 chemistry-2 canon tests."""

from __future__ import annotations

from quant_fund.models.medicinal_chemistry import bench_medicinal_chemistry
from quant_fund.models.photochemistry import bench_photochemistry
from quant_fund.models.quantum_chemistry import bench_quantum_chemistry
from quant_fund.models.spectroscopy import bench_spectroscopy
from quant_fund.models.stereochemistry import bench_stereochemistry
from quant_fund.models.supramolecular_chemistry import bench_supramolecular_chemistry


def test_quantum_chemistry():
    assert bench_quantum_chemistry()["synthetic_quantum_chemistry"] == 1.0


def test_spectroscopy():
    assert bench_spectroscopy()["synthetic_spectroscopy"] == 1.0


def test_photochemistry():
    assert bench_photochemistry()["synthetic_photochemistry"] == 1.0


def test_stereochemistry():
    assert bench_stereochemistry()["synthetic_stereochemistry"] == 1.0


def test_supramolecular_chemistry():
    assert bench_supramolecular_chemistry()["synthetic_supramolecular_chemistry"] == 1.0


def test_medicinal_chemistry():
    assert bench_medicinal_chemistry()["synthetic_medicinal_chemistry"] == 1.0

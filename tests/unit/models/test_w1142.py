"""Wave-1142 quantum-technology canon tests."""

from __future__ import annotations

from quant_fund.models.quantum_chemistry_2 import bench_quantum_chemistry_2
from quant_fund.models.quantum_computing import bench_quantum_computing
from quant_fund.models.quantum_error_2 import bench_quantum_error_2
from quant_fund.models.quantum_information_2 import bench_quantum_information_2
from quant_fund.models.quantum_optics import bench_quantum_optics
from quant_fund.models.quantum_sensing import bench_quantum_sensing


def test_quantum_computing():
    assert bench_quantum_computing()["synthetic_quantum_computing"] == 1.0


def test_quantum_information_2():
    assert bench_quantum_information_2()["synthetic_quantum_information_2"] == 1.0


def test_quantum_chemistry_2():
    assert bench_quantum_chemistry_2()["synthetic_quantum_chemistry_2"] == 1.0


def test_quantum_optics():
    assert bench_quantum_optics()["synthetic_quantum_optics"] == 1.0


def test_quantum_sensing():
    assert bench_quantum_sensing()["synthetic_quantum_sensing"] == 1.0


def test_quantum_error_2():
    assert bench_quantum_error_2()["synthetic_quantum_error_2"] == 1.0

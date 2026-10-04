"""Wave-1212 neurology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.neuro_ophthalmology import bench_neuro_ophthalmology
from quant_fund.models.neurocritical_care import bench_neurocritical_care
from quant_fund.models.neurogenetics import bench_neurogenetics
from quant_fund.models.neuroimmunology import bench_neuroimmunology
from quant_fund.models.neuromuscular_medicine import bench_neuromuscular_medicine
from quant_fund.models.neurovascular_studies import bench_neurovascular_studies


def test_neurocritical_care():
    assert bench_neurocritical_care()["synthetic_neurocritical_care"] == 1.0


def test_neurovascular_studies():
    assert bench_neurovascular_studies()["synthetic_neurovascular_studies"] == 1.0


def test_neuromuscular_medicine():
    assert bench_neuromuscular_medicine()["synthetic_neuromuscular_medicine"] == 1.0


def test_neuro_ophthalmology():
    assert bench_neuro_ophthalmology()["synthetic_neuro_ophthalmology"] == 1.0


def test_neuroimmunology():
    assert bench_neuroimmunology()["synthetic_neuroimmunology"] == 1.0


def test_neurogenetics():
    assert bench_neurogenetics()["synthetic_neurogenetics"] == 1.0

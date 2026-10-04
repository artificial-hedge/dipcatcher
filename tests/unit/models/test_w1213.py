"""Wave-1213 neurology-3 canon tests."""

from __future__ import annotations

from quant_fund.models.neurorehabilitation import bench_neurorehabilitation
from quant_fund.models.neurosurgery_studies import bench_neurosurgery_studies
from quant_fund.models.neurotoxicology import bench_neurotoxicology
from quant_fund.models.neurotrauma import bench_neurotrauma
from quant_fund.models.neurovascular_surgery import bench_neurovascular_surgery
from quant_fund.models.spinal_cord_medicine import bench_spinal_cord_medicine


def test_neurosurgery_studies():
    assert bench_neurosurgery_studies()["synthetic_neurosurgery_studies"] == 1.0


def test_neurotrauma():
    assert bench_neurotrauma()["synthetic_neurotrauma"] == 1.0


def test_neurotoxicology():
    assert bench_neurotoxicology()["synthetic_neurotoxicology"] == 1.0


def test_neurorehabilitation():
    assert bench_neurorehabilitation()["synthetic_neurorehabilitation"] == 1.0


def test_neurovascular_surgery():
    assert bench_neurovascular_surgery()["synthetic_neurovascular_surgery"] == 1.0


def test_spinal_cord_medicine():
    assert bench_spinal_cord_medicine()["synthetic_spinal_cord_medicine"] == 1.0

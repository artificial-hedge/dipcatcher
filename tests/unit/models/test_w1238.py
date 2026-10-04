"""Wave-1238 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.aortic_medicine_studies import bench_aortic_medicine_studies
from quant_fund.models.lymphatic_medicine import bench_lymphatic_medicine
from quant_fund.models.peripheral_artery_studies import bench_peripheral_artery_studies
from quant_fund.models.phlebology_studies import bench_phlebology_studies
from quant_fund.models.vascular_lab_studies import bench_vascular_lab_studies
from quant_fund.models.vascular_medicine_studies import bench_vascular_medicine_studies


def test_vascular_medicine_studies() -> None:
    assert bench_vascular_medicine_studies()["synthetic_vascular_medicine_studies"] == 1.0


def test_phlebology_studies() -> None:
    assert bench_phlebology_studies()["synthetic_phlebology_studies"] == 1.0


def test_lymphatic_medicine() -> None:
    assert bench_lymphatic_medicine()["synthetic_lymphatic_medicine"] == 1.0


def test_vascular_lab_studies() -> None:
    assert bench_vascular_lab_studies()["synthetic_vascular_lab_studies"] == 1.0


def test_peripheral_artery_studies() -> None:
    assert bench_peripheral_artery_studies()["synthetic_peripheral_artery_studies"] == 1.0


def test_aortic_medicine_studies() -> None:
    assert bench_aortic_medicine_studies()["synthetic_aortic_medicine_studies"] == 1.0

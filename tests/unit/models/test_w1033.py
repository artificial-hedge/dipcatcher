"""Wave-1033 biomedical-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.bioinstrumentation import bench_bioinstrumentation
from quant_fund.models.biomechanics import bench_biomechanics
from quant_fund.models.biomedical_imaging2 import bench_biomedical_imaging2
from quant_fund.models.medical_devices import bench_medical_devices
from quant_fund.models.physiological_modeling import bench_physiological_modeling
from quant_fund.models.tissue_engineering import bench_tissue_engineering


def test_biomechanics():
    assert bench_biomechanics()["synthetic_biomechanics"] == 1.0


def test_medical_devices():
    assert bench_medical_devices()["synthetic_medical_devices"] == 1.0


def test_tissue_engineering():
    assert bench_tissue_engineering()["synthetic_tissue_engineering"] == 1.0


def test_bioinstrumentation():
    assert bench_bioinstrumentation()["synthetic_bioinstrumentation"] == 1.0


def test_physiological_modeling():
    assert bench_physiological_modeling()["synthetic_physiological_modeling"] == 1.0


def test_biomedical_imaging2():
    assert bench_biomedical_imaging2()["synthetic_biomedical_imaging2"] == 1.0

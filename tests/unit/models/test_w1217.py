"""Wave-1217 endocrinology canon tests."""

from __future__ import annotations

from quant_fund.models.adrenal_medicine import bench_adrenal_medicine
from quant_fund.models.bone_metabolism import bench_bone_metabolism
from quant_fund.models.diabetes_medicine import bench_diabetes_medicine
from quant_fund.models.endocrinology_studies import bench_endocrinology_studies
from quant_fund.models.metabolic_medicine import bench_metabolic_medicine
from quant_fund.models.thyroid_medicine import bench_thyroid_medicine


def test_endocrinology_studies():
    assert bench_endocrinology_studies()["synthetic_endocrinology_studies"] == 1.0


def test_diabetes_medicine():
    assert bench_diabetes_medicine()["synthetic_diabetes_medicine"] == 1.0


def test_thyroid_medicine():
    assert bench_thyroid_medicine()["synthetic_thyroid_medicine"] == 1.0


def test_metabolic_medicine():
    assert bench_metabolic_medicine()["synthetic_metabolic_medicine"] == 1.0


def test_bone_metabolism():
    assert bench_bone_metabolism()["synthetic_bone_metabolism"] == 1.0


def test_adrenal_medicine():
    assert bench_adrenal_medicine()["synthetic_adrenal_medicine"] == 1.0

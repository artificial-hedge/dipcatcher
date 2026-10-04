"""Wave-1130 anthropology-5 canon tests."""

from __future__ import annotations

from quant_fund.models.anthropology_of_religion import bench_anthropology_of_religion
from quant_fund.models.cognitive_anthropology import bench_cognitive_anthropology
from quant_fund.models.kinship_studies import bench_kinship_studies
from quant_fund.models.material_culture import bench_material_culture
from quant_fund.models.museum_anthropology import bench_museum_anthropology
from quant_fund.models.social_anthropology import bench_social_anthropology


def test_social_anthropology():
    assert bench_social_anthropology()["synthetic_social_anthropology"] == 1.0


def test_cognitive_anthropology():
    assert bench_cognitive_anthropology()["synthetic_cognitive_anthropology"] == 1.0


def test_anthropology_of_religion():
    assert bench_anthropology_of_religion()["synthetic_anthropology_of_religion"] == 1.0


def test_kinship_studies():
    assert bench_kinship_studies()["synthetic_kinship_studies"] == 1.0


def test_material_culture():
    assert bench_material_culture()["synthetic_material_culture"] == 1.0


def test_museum_anthropology():
    assert bench_museum_anthropology()["synthetic_museum_anthropology"] == 1.0

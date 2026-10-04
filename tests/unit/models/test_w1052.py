"""Wave-1052 nutrition canon tests."""

from __future__ import annotations

from quant_fund.models.clinical_nutrition import bench_clinical_nutrition
from quant_fund.models.dietary_assessment import bench_dietary_assessment
from quant_fund.models.metabolic_health import bench_metabolic_health
from quant_fund.models.nutritional_biochemistry import bench_nutritional_biochemistry
from quant_fund.models.nutritional_epidemiology import bench_nutritional_epidemiology
from quant_fund.models.sports_nutrition import bench_sports_nutrition


def test_nutritional_biochemistry():
    assert bench_nutritional_biochemistry()["synthetic_nutritional_biochemistry"] == 1.0


def test_dietary_assessment():
    assert bench_dietary_assessment()["synthetic_dietary_assessment"] == 1.0


def test_clinical_nutrition():
    assert bench_clinical_nutrition()["synthetic_clinical_nutrition"] == 1.0


def test_sports_nutrition():
    assert bench_sports_nutrition()["synthetic_sports_nutrition"] == 1.0


def test_nutritional_epidemiology():
    assert bench_nutritional_epidemiology()["synthetic_nutritional_epidemiology"] == 1.0


def test_metabolic_health():
    assert bench_metabolic_health()["synthetic_metabolic_health"] == 1.0

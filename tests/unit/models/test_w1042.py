"""Wave-1042 food-science canon tests."""

from __future__ import annotations

from quant_fund.models.food_chemistry import bench_food_chemistry
from quant_fund.models.food_microbiology import bench_food_microbiology
from quant_fund.models.food_processing import bench_food_processing
from quant_fund.models.food_safety import bench_food_safety
from quant_fund.models.nutrition_science import bench_nutrition_science
from quant_fund.models.sensory_evaluation import bench_sensory_evaluation


def test_food_chemistry():
    assert bench_food_chemistry()["synthetic_food_chemistry"] == 1.0


def test_food_microbiology():
    assert bench_food_microbiology()["synthetic_food_microbiology"] == 1.0


def test_food_processing():
    assert bench_food_processing()["synthetic_food_processing"] == 1.0


def test_nutrition_science():
    assert bench_nutrition_science()["synthetic_nutrition_science"] == 1.0


def test_sensory_evaluation():
    assert bench_sensory_evaluation()["synthetic_sensory_evaluation"] == 1.0


def test_food_safety():
    assert bench_food_safety()["synthetic_food_safety"] == 1.0

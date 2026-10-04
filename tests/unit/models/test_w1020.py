"""Wave-1020 ecology/evolution canon tests."""

from __future__ import annotations

from quant_fund.models.food_web import bench_food_web
from quant_fund.models.island_biogeography import bench_island_biogeography
from quant_fund.models.logistic_growth import bench_logistic_growth
from quant_fund.models.lotka_volterra import bench_lotka_volterra
from quant_fund.models.neutral_theory import bench_neutral_theory
from quant_fund.models.predator_prey import bench_predator_prey


def test_predator_prey():
    assert bench_predator_prey()["synthetic_predator_prey"] == 1.0


def test_lotka_volterra():
    assert bench_lotka_volterra()["synthetic_lotka_volterra"] == 1.0


def test_logistic_growth():
    assert bench_logistic_growth()["synthetic_logistic_growth"] == 1.0


def test_island_biogeography():
    assert bench_island_biogeography()["synthetic_island_biogeography"] == 1.0


def test_neutral_theory():
    assert bench_neutral_theory()["synthetic_neutral_theory"] == 1.0


def test_food_web():
    assert bench_food_web()["synthetic_food_web"] == 1.0

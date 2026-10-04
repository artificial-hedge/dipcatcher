"""Wave-1211 neurology canon tests."""

from __future__ import annotations

from quant_fund.models.epilepsy_studies import bench_epilepsy_studies
from quant_fund.models.headache_medicine import bench_headache_medicine
from quant_fund.models.movement_disorders import bench_movement_disorders
from quant_fund.models.neurodevelopmental_disorders import bench_neurodevelopmental_disorders
from quant_fund.models.neuropsychiatry_studies import bench_neuropsychiatry_studies
from quant_fund.models.pediatric_neurology import bench_pediatric_neurology


def test_pediatric_neurology():
    assert bench_pediatric_neurology()["synthetic_pediatric_neurology"] == 1.0


def test_neurodevelopmental_disorders():
    assert bench_neurodevelopmental_disorders()["synthetic_neurodevelopmental_disorders"] == 1.0


def test_neuropsychiatry_studies():
    assert bench_neuropsychiatry_studies()["synthetic_neuropsychiatry_studies"] == 1.0


def test_headache_medicine():
    assert bench_headache_medicine()["synthetic_headache_medicine"] == 1.0


def test_epilepsy_studies():
    assert bench_epilepsy_studies()["synthetic_epilepsy_studies"] == 1.0


def test_movement_disorders():
    assert bench_movement_disorders()["synthetic_movement_disorders"] == 1.0

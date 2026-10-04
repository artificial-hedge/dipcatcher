"""Wave-1098 anthropology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.biological_anthropology import bench_biological_anthropology
from quant_fund.models.economic_anthropology import bench_economic_anthropology
from quant_fund.models.medical_anthropology import bench_medical_anthropology
from quant_fund.models.paleoanthropology import bench_paleoanthropology
from quant_fund.models.political_anthropology import bench_political_anthropology
from quant_fund.models.urban_anthropology import bench_urban_anthropology


def test_biological_anthropology():
    assert bench_biological_anthropology()["synthetic_biological_anthropology"] == 1.0


def test_paleoanthropology():
    assert bench_paleoanthropology()["synthetic_paleoanthropology"] == 1.0


def test_medical_anthropology():
    assert bench_medical_anthropology()["synthetic_medical_anthropology"] == 1.0


def test_economic_anthropology():
    assert bench_economic_anthropology()["synthetic_economic_anthropology"] == 1.0


def test_political_anthropology():
    assert bench_political_anthropology()["synthetic_political_anthropology"] == 1.0


def test_urban_anthropology():
    assert bench_urban_anthropology()["synthetic_urban_anthropology"] == 1.0

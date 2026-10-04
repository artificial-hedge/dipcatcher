"""Wave-1055 anthropology canon tests."""

from __future__ import annotations

from quant_fund.models.archaeology import bench_archaeology
from quant_fund.models.cultural_anthropology import bench_cultural_anthropology
from quant_fund.models.ethnography import bench_ethnography
from quant_fund.models.linguistic_anthropology import bench_linguistic_anthropology
from quant_fund.models.physical_anthropology import bench_physical_anthropology
from quant_fund.models.primatology import bench_primatology


def test_physical_anthropology():
    assert bench_physical_anthropology()["synthetic_physical_anthropology"] == 1.0


def test_cultural_anthropology():
    assert bench_cultural_anthropology()["synthetic_cultural_anthropology"] == 1.0


def test_archaeology():
    assert bench_archaeology()["synthetic_archaeology"] == 1.0


def test_linguistic_anthropology():
    assert bench_linguistic_anthropology()["synthetic_linguistic_anthropology"] == 1.0


def test_primatology():
    assert bench_primatology()["synthetic_primatology"] == 1.0


def test_ethnography():
    assert bench_ethnography()["synthetic_ethnography"] == 1.0

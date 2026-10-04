"""Wave-1194 clinical-specialties canon tests."""

from __future__ import annotations

from quant_fund.models.genetic_counseling import bench_genetic_counseling
from quant_fund.models.lactation_consulting import bench_lactation_consulting
from quant_fund.models.perfusion_technology import bench_perfusion_technology
from quant_fund.models.podiatric_medicine import bench_podiatric_medicine
from quant_fund.models.radiation_therapy import bench_radiation_therapy
from quant_fund.models.respiratory_therapy import bench_respiratory_therapy


def test_genetic_counseling():
    assert bench_genetic_counseling()["synthetic_genetic_counseling"] == 1.0


def test_lactation_consulting():
    assert bench_lactation_consulting()["synthetic_lactation_consulting"] == 1.0


def test_podiatric_medicine():
    assert bench_podiatric_medicine()["synthetic_podiatric_medicine"] == 1.0


def test_respiratory_therapy():
    assert bench_respiratory_therapy()["synthetic_respiratory_therapy"] == 1.0


def test_perfusion_technology():
    assert bench_perfusion_technology()["synthetic_perfusion_technology"] == 1.0


def test_radiation_therapy():
    assert bench_radiation_therapy()["synthetic_radiation_therapy"] == 1.0

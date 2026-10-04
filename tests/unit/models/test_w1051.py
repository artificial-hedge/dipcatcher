"""Wave-1051 public-health canon tests."""

from __future__ import annotations

from quant_fund.models.biostatistics_2 import bench_biostatistics_2
from quant_fund.models.epidemiology_2 import bench_epidemiology_2
from quant_fund.models.global_health import bench_global_health
from quant_fund.models.health_policy import bench_health_policy
from quant_fund.models.occupational_health import bench_occupational_health
from quant_fund.models.preventive_medicine import bench_preventive_medicine


def test_epidemiology_2():
    assert bench_epidemiology_2()["synthetic_epidemiology_2"] == 1.0


def test_biostatistics_2():
    assert bench_biostatistics_2()["synthetic_biostatistics_2"] == 1.0


def test_health_policy():
    assert bench_health_policy()["synthetic_health_policy"] == 1.0


def test_global_health():
    assert bench_global_health()["synthetic_global_health"] == 1.0


def test_occupational_health():
    assert bench_occupational_health()["synthetic_occupational_health"] == 1.0


def test_preventive_medicine():
    assert bench_preventive_medicine()["synthetic_preventive_medicine"] == 1.0

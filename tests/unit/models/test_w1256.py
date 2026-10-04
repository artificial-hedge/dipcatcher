"""Wave-1256 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.community_health_studies import bench_community_health_studies
from quant_fund.models.health_disparities_studies import bench_health_disparities_studies
from quant_fund.models.outbreak_studies import bench_outbreak_studies
from quant_fund.models.screening_studies import bench_screening_studies
from quant_fund.models.surveillance_studies import bench_surveillance_studies
from quant_fund.models.vaccination_studies import bench_vaccination_studies


def test_screening_studies() -> None:
    assert bench_screening_studies()["synthetic_screening_studies"] == 1.0


def test_vaccination_studies() -> None:
    assert bench_vaccination_studies()["synthetic_vaccination_studies"] == 1.0


def test_outbreak_studies() -> None:
    assert bench_outbreak_studies()["synthetic_outbreak_studies"] == 1.0


def test_surveillance_studies() -> None:
    assert bench_surveillance_studies()["synthetic_surveillance_studies"] == 1.0


def test_health_disparities_studies() -> None:
    assert bench_health_disparities_studies()["synthetic_health_disparities_studies"] == 1.0


def test_community_health_studies() -> None:
    assert bench_community_health_studies()["synthetic_community_health_studies"] == 1.0

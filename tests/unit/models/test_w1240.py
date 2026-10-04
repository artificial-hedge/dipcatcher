"""Wave-1240 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.asthma_studies import bench_asthma_studies
from quant_fund.models.bronchiectasis_studies import bench_bronchiectasis_studies
from quant_fund.models.copd_studies import bench_copd_studies
from quant_fund.models.interstitial_lung_studies import bench_interstitial_lung_studies
from quant_fund.models.respiratory_studies import bench_respiratory_studies
from quant_fund.models.sleep_breathing_studies import bench_sleep_breathing_studies


def test_respiratory_studies() -> None:
    assert bench_respiratory_studies()["synthetic_respiratory_studies"] == 1.0


def test_asthma_studies() -> None:
    assert bench_asthma_studies()["synthetic_asthma_studies"] == 1.0


def test_copd_studies() -> None:
    assert bench_copd_studies()["synthetic_copd_studies"] == 1.0


def test_interstitial_lung_studies() -> None:
    assert bench_interstitial_lung_studies()["synthetic_interstitial_lung_studies"] == 1.0


def test_sleep_breathing_studies() -> None:
    assert bench_sleep_breathing_studies()["synthetic_sleep_breathing_studies"] == 1.0


def test_bronchiectasis_studies() -> None:
    assert bench_bronchiectasis_studies()["synthetic_bronchiectasis_studies"] == 1.0

"""Wave-1054 sociology canon tests."""

from __future__ import annotations

from quant_fund.models.criminology import bench_criminology
from quant_fund.models.demography import bench_demography
from quant_fund.models.economic_sociology import bench_economic_sociology
from quant_fund.models.social_networks import bench_social_networks
from quant_fund.models.social_stratification import bench_social_stratification
from quant_fund.models.urban_sociology import bench_urban_sociology


def test_social_networks():
    assert bench_social_networks()["synthetic_social_networks"] == 1.0


def test_demography():
    assert bench_demography()["synthetic_demography"] == 1.0


def test_criminology():
    assert bench_criminology()["synthetic_criminology"] == 1.0


def test_urban_sociology():
    assert bench_urban_sociology()["synthetic_urban_sociology"] == 1.0


def test_economic_sociology():
    assert bench_economic_sociology()["synthetic_economic_sociology"] == 1.0


def test_social_stratification():
    assert bench_social_stratification()["synthetic_social_stratification"] == 1.0

"""Wave-1066 area studies canon tests."""

from __future__ import annotations

from quant_fund.models.african_studies import bench_african_studies
from quant_fund.models.asian_studies import bench_asian_studies
from quant_fund.models.european_studies import bench_european_studies
from quant_fund.models.latin_american_studies import bench_latin_american_studies
from quant_fund.models.middle_eastern_studies import bench_middle_eastern_studies
from quant_fund.models.slavic_studies import bench_slavic_studies


def test_latin_american_studies():
    assert bench_latin_american_studies()["synthetic_latin_american_studies"] == 1.0


def test_asian_studies():
    assert bench_asian_studies()["synthetic_asian_studies"] == 1.0


def test_european_studies():
    assert bench_european_studies()["synthetic_european_studies"] == 1.0


def test_middle_eastern_studies():
    assert bench_middle_eastern_studies()["synthetic_middle_eastern_studies"] == 1.0


def test_african_studies():
    assert bench_african_studies()["synthetic_african_studies"] == 1.0


def test_slavic_studies():
    assert bench_slavic_studies()["synthetic_slavic_studies"] == 1.0

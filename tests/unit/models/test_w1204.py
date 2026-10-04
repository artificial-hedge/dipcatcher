"""Wave-1204 geriatric-care canon tests."""

from __future__ import annotations

from quant_fund.models.aging_research import bench_aging_research
from quant_fund.models.geriatric_medicine import bench_geriatric_medicine
from quant_fund.models.gerontology_studies import bench_gerontology_studies
from quant_fund.models.hospice_care import bench_hospice_care
from quant_fund.models.longevity_medicine import bench_longevity_medicine
from quant_fund.models.palliative_care import bench_palliative_care


def test_geriatric_medicine():
    assert bench_geriatric_medicine()["synthetic_geriatric_medicine"] == 1.0


def test_palliative_care():
    assert bench_palliative_care()["synthetic_palliative_care"] == 1.0


def test_hospice_care():
    assert bench_hospice_care()["synthetic_hospice_care"] == 1.0


def test_gerontology_studies():
    assert bench_gerontology_studies()["synthetic_gerontology_studies"] == 1.0


def test_aging_research():
    assert bench_aging_research()["synthetic_aging_research"] == 1.0


def test_longevity_medicine():
    assert bench_longevity_medicine()["synthetic_longevity_medicine"] == 1.0

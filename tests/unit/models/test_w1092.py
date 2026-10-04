"""Wave-1092 art-historiography canon tests."""

from __future__ import annotations

from quant_fund.models.connoisseurship import bench_connoisseurship
from quant_fund.models.curation_practice import bench_curation_practice
from quant_fund.models.formal_analysis import bench_formal_analysis
from quant_fund.models.iconography import bench_iconography
from quant_fund.models.iconology import bench_iconology
from quant_fund.models.provenance_studies import bench_provenance_studies


def test_iconography():
    assert bench_iconography()["synthetic_iconography"] == 1.0


def test_iconology():
    assert bench_iconology()["synthetic_iconology"] == 1.0


def test_connoisseurship():
    assert bench_connoisseurship()["synthetic_connoisseurship"] == 1.0


def test_provenance_studies():
    assert bench_provenance_studies()["synthetic_provenance_studies"] == 1.0


def test_curation_practice():
    assert bench_curation_practice()["synthetic_curation_practice"] == 1.0


def test_formal_analysis():
    assert bench_formal_analysis()["synthetic_formal_analysis"] == 1.0

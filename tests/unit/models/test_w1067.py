"""Wave-1067 library/information science canon tests."""

from __future__ import annotations

from quant_fund.models.archival_studies import bench_archival_studies
from quant_fund.models.digital_humanities import bench_digital_humanities
from quant_fund.models.information_science import bench_information_science
from quant_fund.models.knowledge_organization import bench_knowledge_organization
from quant_fund.models.library_science import bench_library_science
from quant_fund.models.museum_studies import bench_museum_studies


def test_library_science():
    assert bench_library_science()["synthetic_library_science"] == 1.0


def test_information_science():
    assert bench_information_science()["synthetic_information_science"] == 1.0


def test_archival_studies():
    assert bench_archival_studies()["synthetic_archival_studies"] == 1.0


def test_museum_studies():
    assert bench_museum_studies()["synthetic_museum_studies"] == 1.0


def test_digital_humanities():
    assert bench_digital_humanities()["synthetic_digital_humanities"] == 1.0


def test_knowledge_organization():
    assert bench_knowledge_organization()["synthetic_knowledge_organization"] == 1.0

"""Wave-1090 history-of-science canon tests."""

from __future__ import annotations

from quant_fund.models.history_of_science import bench_history_of_science
from quant_fund.models.information_history import bench_information_history
from quant_fund.models.media_archaeology import bench_media_archaeology
from quant_fund.models.philosophy_of_technology import bench_philosophy_of_technology
from quant_fund.models.sts_studies import bench_sts_studies
from quant_fund.models.technology_studies import bench_technology_studies


def test_history_of_science():
    assert bench_history_of_science()["synthetic_history_of_science"] == 1.0


def test_sts_studies():
    assert bench_sts_studies()["synthetic_sts_studies"] == 1.0


def test_philosophy_of_technology():
    assert bench_philosophy_of_technology()["synthetic_philosophy_of_technology"] == 1.0


def test_media_archaeology():
    assert bench_media_archaeology()["synthetic_media_archaeology"] == 1.0


def test_information_history():
    assert bench_information_history()["synthetic_information_history"] == 1.0


def test_technology_studies():
    assert bench_technology_studies()["synthetic_technology_studies"] == 1.0

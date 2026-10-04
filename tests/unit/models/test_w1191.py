"""Wave-1191 media canon tests."""

from __future__ import annotations

from quant_fund.models.advertising_studies import bench_advertising_studies
from quant_fund.models.broadcasting_studies import bench_broadcasting_studies
from quant_fund.models.journalism_studies import bench_journalism_studies
from quant_fund.models.news_media import bench_news_media
from quant_fund.models.public_relations_studies import bench_public_relations_studies
from quant_fund.models.publishing_studies import bench_publishing_studies


def test_journalism_studies():
    assert bench_journalism_studies()["synthetic_journalism_studies"] == 1.0


def test_advertising_studies():
    assert bench_advertising_studies()["synthetic_advertising_studies"] == 1.0


def test_broadcasting_studies():
    assert bench_broadcasting_studies()["synthetic_broadcasting_studies"] == 1.0


def test_news_media():
    assert bench_news_media()["synthetic_news_media"] == 1.0


def test_public_relations_studies():
    assert bench_public_relations_studies()["synthetic_public_relations_studies"] == 1.0


def test_publishing_studies():
    assert bench_publishing_studies()["synthetic_publishing_studies"] == 1.0

"""Wave-1126 psychology-4 canon tests."""

from __future__ import annotations

from quant_fund.models.community_psychology import bench_community_psychology
from quant_fund.models.consumer_psychology import bench_consumer_psychology
from quant_fund.models.cross_cultural_psychology import bench_cross_cultural_psychology
from quant_fund.models.political_psychology import bench_political_psychology
from quant_fund.models.positive_psychology import bench_positive_psychology
from quant_fund.models.social_cognition import bench_social_cognition


def test_social_cognition():
    assert bench_social_cognition()["synthetic_social_cognition"] == 1.0


def test_positive_psychology():
    assert bench_positive_psychology()["synthetic_positive_psychology"] == 1.0


def test_cross_cultural_psychology():
    assert bench_cross_cultural_psychology()["synthetic_cross_cultural_psychology"] == 1.0


def test_consumer_psychology():
    assert bench_consumer_psychology()["synthetic_consumer_psychology"] == 1.0


def test_political_psychology():
    assert bench_political_psychology()["synthetic_political_psychology"] == 1.0


def test_community_psychology():
    assert bench_community_psychology()["synthetic_community_psychology"] == 1.0

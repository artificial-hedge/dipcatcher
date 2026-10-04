"""Wave-1082 jewish studies canon tests."""

from __future__ import annotations

from quant_fund.models.hebrew_language import bench_hebrew_language
from quant_fund.models.jewish_philosophy import bench_jewish_philosophy
from quant_fund.models.jewish_studies import bench_jewish_studies
from quant_fund.models.kabbalah import bench_kabbalah
from quant_fund.models.rabbinics import bench_rabbinics
from quant_fund.models.talmudic_studies import bench_talmudic_studies


def test_jewish_studies():
    assert bench_jewish_studies()["synthetic_jewish_studies"] == 1.0


def test_talmudic_studies():
    assert bench_talmudic_studies()["synthetic_talmudic_studies"] == 1.0


def test_hebrew_language():
    assert bench_hebrew_language()["synthetic_hebrew_language"] == 1.0


def test_rabbinics():
    assert bench_rabbinics()["synthetic_rabbinics"] == 1.0


def test_kabbalah():
    assert bench_kabbalah()["synthetic_kabbalah"] == 1.0


def test_jewish_philosophy():
    assert bench_jewish_philosophy()["synthetic_jewish_philosophy"] == 1.0

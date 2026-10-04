"""Wave-1089 linguistics-2 canon tests."""

from __future__ import annotations

from quant_fund.models.computational_linguistics import bench_computational_linguistics
from quant_fund.models.corpus_linguistics import bench_corpus_linguistics
from quant_fund.models.dialectology import bench_dialectology
from quant_fund.models.historical_linguistics import bench_historical_linguistics
from quant_fund.models.psycholinguistics import bench_psycholinguistics
from quant_fund.models.sociolinguistics import bench_sociolinguistics


def test_sociolinguistics():
    assert bench_sociolinguistics()["synthetic_sociolinguistics"] == 1.0


def test_psycholinguistics():
    assert bench_psycholinguistics()["synthetic_psycholinguistics"] == 1.0


def test_computational_linguistics():
    assert bench_computational_linguistics()["synthetic_computational_linguistics"] == 1.0


def test_corpus_linguistics():
    assert bench_corpus_linguistics()["synthetic_corpus_linguistics"] == 1.0


def test_dialectology():
    assert bench_dialectology()["synthetic_dialectology"] == 1.0


def test_historical_linguistics():
    assert bench_historical_linguistics()["synthetic_historical_linguistics"] == 1.0

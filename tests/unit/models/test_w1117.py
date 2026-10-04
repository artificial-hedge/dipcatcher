"""Wave-1117 sociology-3 canon tests."""

from __future__ import annotations

from quant_fund.models.historical_sociology import bench_historical_sociology
from quant_fund.models.legal_sociology import bench_legal_sociology
from quant_fund.models.mathematical_sociology import bench_mathematical_sociology
from quant_fund.models.military_sociology import bench_military_sociology
from quant_fund.models.science_studies import bench_science_studies
from quant_fund.models.sociology_of_knowledge import bench_sociology_of_knowledge


def test_mathematical_sociology():
    assert bench_mathematical_sociology()["synthetic_mathematical_sociology"] == 1.0


def test_historical_sociology():
    assert bench_historical_sociology()["synthetic_historical_sociology"] == 1.0


def test_science_studies():
    assert bench_science_studies()["synthetic_science_studies"] == 1.0


def test_sociology_of_knowledge():
    assert bench_sociology_of_knowledge()["synthetic_sociology_of_knowledge"] == 1.0


def test_military_sociology():
    assert bench_military_sociology()["synthetic_military_sociology"] == 1.0


def test_legal_sociology():
    assert bench_legal_sociology()["synthetic_legal_sociology"] == 1.0

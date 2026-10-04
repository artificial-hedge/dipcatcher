"""Wave-1062 religious-studies canon tests."""

from __future__ import annotations

from quant_fund.models.biblical_studies import bench_biblical_studies
from quant_fund.models.buddhist_studies import bench_buddhist_studies
from quant_fund.models.comparative_religion import bench_comparative_religion
from quant_fund.models.islamic_studies import bench_islamic_studies
from quant_fund.models.religious_ethics import bench_religious_ethics
from quant_fund.models.theology import bench_theology


def test_theology():
    assert bench_theology()["synthetic_theology"] == 1.0


def test_comparative_religion():
    assert bench_comparative_religion()["synthetic_comparative_religion"] == 1.0


def test_biblical_studies():
    assert bench_biblical_studies()["synthetic_biblical_studies"] == 1.0


def test_islamic_studies():
    assert bench_islamic_studies()["synthetic_islamic_studies"] == 1.0


def test_buddhist_studies():
    assert bench_buddhist_studies()["synthetic_buddhist_studies"] == 1.0


def test_religious_ethics():
    assert bench_religious_ethics()["synthetic_religious_ethics"] == 1.0

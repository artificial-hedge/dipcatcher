"""Wave-1081 renaissance/early modern canon tests."""

from __future__ import annotations

from quant_fund.models.baroque_studies import bench_baroque_studies
from quant_fund.models.early_modern import bench_early_modern
from quant_fund.models.enlightenment_studies import bench_enlightenment_studies
from quant_fund.models.humanism import bench_humanism
from quant_fund.models.reformation_studies import bench_reformation_studies
from quant_fund.models.renaissance_studies import bench_renaissance_studies


def test_renaissance_studies():
    assert bench_renaissance_studies()["synthetic_renaissance_studies"] == 1.0


def test_early_modern():
    assert bench_early_modern()["synthetic_early_modern"] == 1.0


def test_humanism():
    assert bench_humanism()["synthetic_humanism"] == 1.0


def test_reformation_studies():
    assert bench_reformation_studies()["synthetic_reformation_studies"] == 1.0


def test_baroque_studies():
    assert bench_baroque_studies()["synthetic_baroque_studies"] == 1.0


def test_enlightenment_studies():
    assert bench_enlightenment_studies()["synthetic_enlightenment_studies"] == 1.0

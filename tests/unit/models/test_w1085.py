"""Wave-1085 literary-periods canon tests."""

from __future__ import annotations

from quant_fund.models.medieval_literature import bench_medieval_literature
from quant_fund.models.modernism import bench_modernism
from quant_fund.models.postmodernism import bench_postmodernism
from quant_fund.models.renaissance_literature import bench_renaissance_literature
from quant_fund.models.romanticism import bench_romanticism
from quant_fund.models.victorian_studies import bench_victorian_studies


def test_medieval_literature():
    assert bench_medieval_literature()["synthetic_medieval_literature"] == 1.0


def test_renaissance_literature():
    assert bench_renaissance_literature()["synthetic_renaissance_literature"] == 1.0


def test_romanticism():
    assert bench_romanticism()["synthetic_romanticism"] == 1.0


def test_modernism():
    assert bench_modernism()["synthetic_modernism"] == 1.0


def test_postmodernism():
    assert bench_postmodernism()["synthetic_postmodernism"] == 1.0


def test_victorian_studies():
    assert bench_victorian_studies()["synthetic_victorian_studies"] == 1.0

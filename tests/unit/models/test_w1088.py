"""Wave-1088 philosophy-2 canon tests."""

from __future__ import annotations

from quant_fund.models.analytic_philosophy import bench_analytic_philosophy
from quant_fund.models.ancient_philosophy import bench_ancient_philosophy
from quant_fund.models.continental_philosophy import bench_continental_philosophy
from quant_fund.models.existentialism import bench_existentialism
from quant_fund.models.medieval_philosophy import bench_medieval_philosophy
from quant_fund.models.pragmatism import bench_pragmatism


def test_ancient_philosophy():
    assert bench_ancient_philosophy()["synthetic_ancient_philosophy"] == 1.0


def test_medieval_philosophy():
    assert bench_medieval_philosophy()["synthetic_medieval_philosophy"] == 1.0


def test_continental_philosophy():
    assert bench_continental_philosophy()["synthetic_continental_philosophy"] == 1.0


def test_analytic_philosophy():
    assert bench_analytic_philosophy()["synthetic_analytic_philosophy"] == 1.0


def test_pragmatism():
    assert bench_pragmatism()["synthetic_pragmatism"] == 1.0


def test_existentialism():
    assert bench_existentialism()["synthetic_existentialism"] == 1.0

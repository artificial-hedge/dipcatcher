"""Wave-1101 biology canon tests."""

from __future__ import annotations

from quant_fund.models.botany import bench_botany
from quant_fund.models.cell_biology import bench_cell_biology
from quant_fund.models.genetics import bench_genetics
from quant_fund.models.microbiology import bench_microbiology
from quant_fund.models.molecular_biology import bench_molecular_biology
from quant_fund.models.zoology import bench_zoology


def test_molecular_biology():
    assert bench_molecular_biology()["synthetic_molecular_biology"] == 1.0


def test_cell_biology():
    assert bench_cell_biology()["synthetic_cell_biology"] == 1.0


def test_genetics():
    assert bench_genetics()["synthetic_genetics"] == 1.0


def test_microbiology():
    assert bench_microbiology()["synthetic_microbiology"] == 1.0


def test_zoology():
    assert bench_zoology()["synthetic_zoology"] == 1.0


def test_botany():
    assert bench_botany()["synthetic_botany"] == 1.0

"""Wave-1152 molecular-life-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.biochemistry_2 import bench_biochemistry_2
from quant_fund.models.cell_biology_2 import bench_cell_biology_2
from quant_fund.models.genetics_2 import bench_genetics_2
from quant_fund.models.molecular_biology_2 import bench_molecular_biology_2
from quant_fund.models.pharmacology_2 import bench_pharmacology_2
from quant_fund.models.toxicology_3 import bench_toxicology_3


def test_biochemistry_2():
    assert bench_biochemistry_2()["synthetic_biochemistry_2"] == 1.0


def test_molecular_biology_2():
    assert bench_molecular_biology_2()["synthetic_molecular_biology_2"] == 1.0


def test_cell_biology_2():
    assert bench_cell_biology_2()["synthetic_cell_biology_2"] == 1.0


def test_genetics_2():
    assert bench_genetics_2()["synthetic_genetics_2"] == 1.0


def test_pharmacology_2():
    assert bench_pharmacology_2()["synthetic_pharmacology_2"] == 1.0


def test_toxicology_3():
    assert bench_toxicology_3()["synthetic_toxicology_3"] == 1.0

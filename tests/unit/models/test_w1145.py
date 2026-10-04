"""Wave-1145 life-science canon tests."""

from __future__ import annotations

from quant_fund.models.bioinformatics_4 import bench_bioinformatics_4
from quant_fund.models.epidemiology_3 import bench_epidemiology_3
from quant_fund.models.genomicsciences import bench_genomicsciences
from quant_fund.models.proteomics import bench_proteomics
from quant_fund.models.synthetic_biology import bench_synthetic_biology
from quant_fund.models.systems_biology_2 import bench_systems_biology_2


def test_genomicsciences():
    assert bench_genomicsciences()["synthetic_genomicsciences"] == 1.0


def test_proteomics():
    assert bench_proteomics()["synthetic_proteomics"] == 1.0


def test_bioinformatics_4():
    assert bench_bioinformatics_4()["synthetic_bioinformatics_4"] == 1.0


def test_systems_biology_2():
    assert bench_systems_biology_2()["synthetic_systems_biology_2"] == 1.0


def test_synthetic_biology():
    assert bench_synthetic_biology()["synthetic_synthetic_biology"] == 1.0


def test_epidemiology_3():
    assert bench_epidemiology_3()["synthetic_epidemiology_3"] == 1.0

"""Wave-1146 organismal-biology canon tests."""

from __future__ import annotations

from quant_fund.models.entomology_2 import bench_entomology_2
from quant_fund.models.limnology import bench_limnology
from quant_fund.models.mycology import bench_mycology
from quant_fund.models.parasitology import bench_parasitology
from quant_fund.models.virology import bench_virology
from quant_fund.models.wildlife_biology import bench_wildlife_biology


def test_virology():
    assert bench_virology()["synthetic_virology"] == 1.0


def test_parasitology():
    assert bench_parasitology()["synthetic_parasitology"] == 1.0


def test_mycology():
    assert bench_mycology()["synthetic_mycology"] == 1.0


def test_entomology_2():
    assert bench_entomology_2()["synthetic_entomology_2"] == 1.0


def test_limnology():
    assert bench_limnology()["synthetic_limnology"] == 1.0


def test_wildlife_biology():
    assert bench_wildlife_biology()["synthetic_wildlife_biology"] == 1.0

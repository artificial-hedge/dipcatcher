"""Wave-1150 microbial-genetics canon tests."""

from __future__ import annotations

from quant_fund.models.bacteriology import bench_bacteriology
from quant_fund.models.epigenetics import bench_epigenetics
from quant_fund.models.immunogenetics import bench_immunogenetics
from quant_fund.models.microbiology_2 import bench_microbiology_2
from quant_fund.models.molecular_genetics import bench_molecular_genetics
from quant_fund.models.virology_2 import bench_virology_2


def test_microbiology_2():
    assert bench_microbiology_2()["synthetic_microbiology_2"] == 1.0


def test_bacteriology():
    assert bench_bacteriology()["synthetic_bacteriology"] == 1.0


def test_virology_2():
    assert bench_virology_2()["synthetic_virology_2"] == 1.0


def test_immunogenetics():
    assert bench_immunogenetics()["synthetic_immunogenetics"] == 1.0


def test_molecular_genetics():
    assert bench_molecular_genetics()["synthetic_molecular_genetics"] == 1.0


def test_epigenetics():
    assert bench_epigenetics()["synthetic_epigenetics"] == 1.0

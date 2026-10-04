"""Wave-1140 earth-science canon tests."""

from __future__ import annotations

from quant_fund.models.glaciology import bench_glaciology
from quant_fund.models.hydrology_2 import bench_hydrology_2
from quant_fund.models.oceanography import bench_oceanography
from quant_fund.models.paleoclimatology import bench_paleoclimatology
from quant_fund.models.seismology import bench_seismology
from quant_fund.models.volcanology_2 import bench_volcanology_2


def test_oceanography():
    assert bench_oceanography()["synthetic_oceanography"] == 1.0


def test_hydrology_2():
    assert bench_hydrology_2()["synthetic_hydrology_2"] == 1.0


def test_seismology():
    assert bench_seismology()["synthetic_seismology"] == 1.0


def test_glaciology():
    assert bench_glaciology()["synthetic_glaciology"] == 1.0


def test_paleoclimatology():
    assert bench_paleoclimatology()["synthetic_paleoclimatology"] == 1.0


def test_volcanology_2():
    assert bench_volcanology_2()["synthetic_volcanology_2"] == 1.0

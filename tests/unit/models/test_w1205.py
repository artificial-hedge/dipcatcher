"""Wave-1205 extreme-environment canon tests."""

from __future__ import annotations

from quant_fund.models.aerospace_medicine import bench_aerospace_medicine
from quant_fund.models.diving_medicine import bench_diving_medicine
from quant_fund.models.high_altitude_medicine import bench_high_altitude_medicine
from quant_fund.models.hyperbaric_oxygen import bench_hyperbaric_oxygen
from quant_fund.models.space_physiology import bench_space_physiology
from quant_fund.models.wilderness_medicine import bench_wilderness_medicine


def test_aerospace_medicine():
    assert bench_aerospace_medicine()["synthetic_aerospace_medicine"] == 1.0


def test_diving_medicine():
    assert bench_diving_medicine()["synthetic_diving_medicine"] == 1.0


def test_wilderness_medicine():
    assert bench_wilderness_medicine()["synthetic_wilderness_medicine"] == 1.0


def test_space_physiology():
    assert bench_space_physiology()["synthetic_space_physiology"] == 1.0


def test_hyperbaric_oxygen():
    assert bench_hyperbaric_oxygen()["synthetic_hyperbaric_oxygen"] == 1.0


def test_high_altitude_medicine():
    assert bench_high_altitude_medicine()["synthetic_high_altitude_medicine"] == 1.0

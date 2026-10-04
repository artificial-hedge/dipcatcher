"""Wave-1019 geophysics-3 canon tests."""

from __future__ import annotations

from quant_fund.models.earthquake_magnitude import bench_earthquake_magnitude
from quant_fund.models.geomagnetism import bench_geomagnetism
from quant_fund.models.gravity_anomaly import bench_gravity_anomaly
from quant_fund.models.heat_flow_geo import bench_heat_flow_geo
from quant_fund.models.plate_tectonics import bench_plate_tectonics
from quant_fund.models.seismic_waves import bench_seismic_waves


def test_seismic_waves():
    assert bench_seismic_waves()["synthetic_seismic_waves"] == 1.0


def test_earthquake_magnitude():
    assert bench_earthquake_magnitude()["synthetic_earthquake_magnitude"] == 1.0


def test_plate_tectonics():
    assert bench_plate_tectonics()["synthetic_plate_tectonics"] == 1.0


def test_gravity_anomaly():
    assert bench_gravity_anomaly()["synthetic_gravity_anomaly"] == 1.0


def test_geomagnetism():
    assert bench_geomagnetism()["synthetic_geomagnetism"] == 1.0


def test_heat_flow_geo():
    assert bench_heat_flow_geo()["synthetic_heat_flow_geo"] == 1.0

"""Wave-1018 relativity-2 canon tests."""

from __future__ import annotations

from quant_fund.models.four_vectors import bench_four_vectors
from quant_fund.models.geodesic_motion import bench_geodesic_motion
from quant_fund.models.gravitational_lensing import bench_gravitational_lensing
from quant_fund.models.gravitational_waves import bench_gravitational_waves
from quant_fund.models.lorentz_transformation import bench_lorentz_transformation
from quant_fund.models.spacetime_interval import bench_spacetime_interval


def test_lorentz_transformation():
    assert bench_lorentz_transformation()["synthetic_lorentz_transformation"] == 1.0


def test_spacetime_interval():
    assert bench_spacetime_interval()["synthetic_spacetime_interval"] == 1.0


def test_four_vectors():
    assert bench_four_vectors()["synthetic_four_vectors"] == 1.0


def test_geodesic_motion():
    assert bench_geodesic_motion()["synthetic_geodesic_motion"] == 1.0


def test_gravitational_lensing():
    assert bench_gravitational_lensing()["synthetic_gravitational_lensing"] == 1.0


def test_gravitational_waves():
    assert bench_gravitational_waves()["synthetic_gravitational_waves"] == 1.0

"""Wave-1003 MHD/plasma canon tests."""

from __future__ import annotations

from quant_fund.models.alfven_waves import bench_alfven_waves
from quant_fund.models.elsaesser_vars import bench_elsaesser_vars
from quant_fund.models.frozen_flux import bench_frozen_flux
from quant_fund.models.magnetic_reconnection import bench_magnetic_reconnection
from quant_fund.models.mhd_equations import bench_mhd_equations
from quant_fund.models.parker_solar_wind import bench_parker_solar_wind


def test_mhd_equations():
    assert bench_mhd_equations()["synthetic_mhd_equations"] == 1.0


def test_alfven_waves():
    assert bench_alfven_waves()["synthetic_alfven_waves"] == 1.0


def test_parker_solar_wind():
    assert bench_parker_solar_wind()["synthetic_parker_solar_wind"] == 1.0


def test_magnetic_reconnection():
    assert bench_magnetic_reconnection()["synthetic_magnetic_reconnection"] == 1.0


def test_frozen_flux():
    assert bench_frozen_flux()["synthetic_frozen_flux"] == 1.0


def test_elsaesser_vars():
    assert bench_elsaesser_vars()["synthetic_elsaesser_vars"] == 1.0

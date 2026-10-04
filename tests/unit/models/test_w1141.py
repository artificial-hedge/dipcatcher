"""Wave-1141 astronomy-4 canon tests."""

from __future__ import annotations

from quant_fund.models.astrobiology import bench_astrobiology
from quant_fund.models.astrochemistry import bench_astrochemistry
from quant_fund.models.cosmology_2 import bench_cosmology_2
from quant_fund.models.exoplanet_science import bench_exoplanet_science
from quant_fund.models.galactic_dynamics import bench_galactic_dynamics
from quant_fund.models.helio_seismology import bench_helio_seismology


def test_cosmology_2():
    assert bench_cosmology_2()["synthetic_cosmology_2"] == 1.0


def test_astrobiology():
    assert bench_astrobiology()["synthetic_astrobiology"] == 1.0


def test_astrochemistry():
    assert bench_astrochemistry()["synthetic_astrochemistry"] == 1.0


def test_helio_seismology():
    assert bench_helio_seismology()["synthetic_helio_seismology"] == 1.0


def test_exoplanet_science():
    assert bench_exoplanet_science()["synthetic_exoplanet_science"] == 1.0


def test_galactic_dynamics():
    assert bench_galactic_dynamics()["synthetic_galactic_dynamics"] == 1.0

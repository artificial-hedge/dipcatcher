"""Wave-1014 astrophysics/cosmology canon tests."""

from __future__ import annotations

from quant_fund.models.cmb_anisotropy import bench_cmb_anisotropy
from quant_fund.models.dark_matter import bench_dark_matter
from quant_fund.models.hubble_law import bench_hubble_law
from quant_fund.models.jeans_instability import bench_jeans_instability
from quant_fund.models.stellar_evolution import bench_stellar_evolution
from quant_fund.models.stellar_structure import bench_stellar_structure


def test_jeans_instability():
    assert bench_jeans_instability()["synthetic_jeans_instability"] == 1.0


def test_stellar_structure():
    assert bench_stellar_structure()["synthetic_stellar_structure"] == 1.0


def test_stellar_evolution():
    assert bench_stellar_evolution()["synthetic_stellar_evolution"] == 1.0


def test_hubble_law():
    assert bench_hubble_law()["synthetic_hubble_law"] == 1.0


def test_cmb_anisotropy():
    assert bench_cmb_anisotropy()["synthetic_cmb_anisotropy"] == 1.0


def test_dark_matter():
    assert bench_dark_matter()["synthetic_dark_matter"] == 1.0

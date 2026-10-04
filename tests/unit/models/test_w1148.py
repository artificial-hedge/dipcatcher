"""Wave-1148 geological-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.geochronology_2 import bench_geochronology_2
from quant_fund.models.geology_3 import bench_geology_3
from quant_fund.models.geomorphology_2 import bench_geomorphology_2
from quant_fund.models.mineralogy_2 import bench_mineralogy_2
from quant_fund.models.petrology_2 import bench_petrology_2
from quant_fund.models.stratigraphy_2 import bench_stratigraphy_2


def test_geology_3():
    assert bench_geology_3()["synthetic_geology_3"] == 1.0


def test_petrology_2():
    assert bench_petrology_2()["synthetic_petrology_2"] == 1.0


def test_mineralogy_2():
    assert bench_mineralogy_2()["synthetic_mineralogy_2"] == 1.0


def test_stratigraphy_2():
    assert bench_stratigraphy_2()["synthetic_stratigraphy_2"] == 1.0


def test_geomorphology_2():
    assert bench_geomorphology_2()["synthetic_geomorphology_2"] == 1.0


def test_geochronology_2():
    assert bench_geochronology_2()["synthetic_geochronology_2"] == 1.0

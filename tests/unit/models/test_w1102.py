"""Wave-1102 geology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.geophysics_applied import bench_geophysics_applied
from quant_fund.models.hydrogeology import bench_hydrogeology
from quant_fund.models.mineralogy import bench_mineralogy
from quant_fund.models.sedimentology import bench_sedimentology
from quant_fund.models.tectonics import bench_tectonics
from quant_fund.models.volcanology import bench_volcanology


def test_mineralogy():
    assert bench_mineralogy()["synthetic_mineralogy"] == 1.0


def test_volcanology():
    assert bench_volcanology()["synthetic_volcanology"] == 1.0


def test_sedimentology():
    assert bench_sedimentology()["synthetic_sedimentology"] == 1.0


def test_tectonics():
    assert bench_tectonics()["synthetic_tectonics"] == 1.0


def test_hydrogeology():
    assert bench_hydrogeology()["synthetic_hydrogeology"] == 1.0


def test_geophysics_applied():
    assert bench_geophysics_applied()["synthetic_geophysics_applied"] == 1.0

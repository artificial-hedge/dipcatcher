"""Wave-1047 marine-biology canon tests."""

from __future__ import annotations

from quant_fund.models.aquaculture import bench_aquaculture
from quant_fund.models.benthic_biology import bench_benthic_biology
from quant_fund.models.coral_reef_ecology import bench_coral_reef_ecology
from quant_fund.models.fisheries_science import bench_fisheries_science
from quant_fund.models.marine_ecology import bench_marine_ecology
from quant_fund.models.plankton_dynamics import bench_plankton_dynamics


def test_plankton_dynamics():
    assert bench_plankton_dynamics()["synthetic_plankton_dynamics"] == 1.0


def test_marine_ecology():
    assert bench_marine_ecology()["synthetic_marine_ecology"] == 1.0


def test_fisheries_science():
    assert bench_fisheries_science()["synthetic_fisheries_science"] == 1.0


def test_aquaculture():
    assert bench_aquaculture()["synthetic_aquaculture"] == 1.0


def test_benthic_biology():
    assert bench_benthic_biology()["synthetic_benthic_biology"] == 1.0


def test_coral_reef_ecology():
    assert bench_coral_reef_ecology()["synthetic_coral_reef_ecology"] == 1.0

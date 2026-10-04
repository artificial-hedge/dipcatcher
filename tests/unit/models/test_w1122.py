"""Wave-1122 archaeology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.archaeogenetics import bench_archaeogenetics
from quant_fund.models.ceramic_analysis import bench_ceramic_analysis
from quant_fund.models.geoarchaeology import bench_geoarchaeology
from quant_fund.models.lithic_analysis import bench_lithic_analysis
from quant_fund.models.paleoethnobotany import bench_paleoethnobotany
from quant_fund.models.zooarchaeology import bench_zooarchaeology


def test_geoarchaeology():
    assert bench_geoarchaeology()["synthetic_geoarchaeology"] == 1.0


def test_zooarchaeology():
    assert bench_zooarchaeology()["synthetic_zooarchaeology"] == 1.0


def test_paleoethnobotany():
    assert bench_paleoethnobotany()["synthetic_paleoethnobotany"] == 1.0


def test_ceramic_analysis():
    assert bench_ceramic_analysis()["synthetic_ceramic_analysis"] == 1.0


def test_lithic_analysis():
    assert bench_lithic_analysis()["synthetic_lithic_analysis"] == 1.0


def test_archaeogenetics():
    assert bench_archaeogenetics()["synthetic_archaeogenetics"] == 1.0

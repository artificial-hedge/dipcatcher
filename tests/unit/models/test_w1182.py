"""Wave-1182 dance canon tests."""

from __future__ import annotations

from quant_fund.models.ballet_studies import bench_ballet_studies
from quant_fund.models.choreography_2 import bench_choreography_2
from quant_fund.models.dance_pedagogy import bench_dance_pedagogy
from quant_fund.models.dance_science import bench_dance_science
from quant_fund.models.movement_studies import bench_movement_studies
from quant_fund.models.somatic_practices import bench_somatic_practices


def test_ballet_studies():
    assert bench_ballet_studies()["synthetic_ballet_studies"] == 1.0


def test_choreography_2():
    assert bench_choreography_2()["synthetic_choreography_2"] == 1.0


def test_dance_pedagogy():
    assert bench_dance_pedagogy()["synthetic_dance_pedagogy"] == 1.0


def test_somatic_practices():
    assert bench_somatic_practices()["synthetic_somatic_practices"] == 1.0


def test_dance_science():
    assert bench_dance_science()["synthetic_dance_science"] == 1.0


def test_movement_studies():
    assert bench_movement_studies()["synthetic_movement_studies"] == 1.0

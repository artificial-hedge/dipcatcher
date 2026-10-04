"""Wave-1181 performing-arts canon tests."""

from __future__ import annotations

from quant_fund.models.acting_studies import bench_acting_studies
from quant_fund.models.directing_studies import bench_directing_studies
from quant_fund.models.performing_arts_2 import bench_performing_arts_2
from quant_fund.models.playwriting import bench_playwriting
from quant_fund.models.scenography import bench_scenography
from quant_fund.models.theater_arts import bench_theater_arts


def test_performing_arts_2():
    assert bench_performing_arts_2()["synthetic_performing_arts_2"] == 1.0


def test_theater_arts():
    assert bench_theater_arts()["synthetic_theater_arts"] == 1.0


def test_acting_studies():
    assert bench_acting_studies()["synthetic_acting_studies"] == 1.0


def test_directing_studies():
    assert bench_directing_studies()["synthetic_directing_studies"] == 1.0


def test_playwriting():
    assert bench_playwriting()["synthetic_playwriting"] == 1.0


def test_scenography():
    assert bench_scenography()["synthetic_scenography"] == 1.0

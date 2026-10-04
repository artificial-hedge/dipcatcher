"""Wave-985 Hardy-space/BMO canon tests."""

from __future__ import annotations

from quant_fund.models.atomic_h1 import bench_atomic_h1
from quant_fund.models.bmo_space import bench_bmo_space
from quant_fund.models.carleson_measure import bench_carleson_measure
from quant_fund.models.fefferman_stein import bench_fefferman_stein
from quant_fund.models.hardy_h1 import bench_hardy_h1
from quant_fund.models.john_nirenberg import bench_john_nirenberg


def test_hardy_h1():
    assert bench_hardy_h1()["synthetic_hardy_h1"] == 1.0


def test_bmo_space():
    assert bench_bmo_space()["synthetic_bmo_space"] == 1.0


def test_atomic_h1():
    assert bench_atomic_h1()["synthetic_atomic_h1"] == 1.0


def test_carleson_measure():
    assert bench_carleson_measure()["synthetic_carleson_measure"] == 1.0


def test_john_nirenberg():
    assert bench_john_nirenberg()["synthetic_john_nirenberg"] == 1.0


def test_fefferman_stein():
    assert bench_fefferman_stein()["synthetic_fefferman_stein"] == 1.0

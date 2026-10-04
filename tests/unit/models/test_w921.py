"""Wave-921 distributed-systems-5 canon tests."""

from __future__ import annotations

from quant_fund.models.atomic_bcast import bench_atomic_bcast
from quant_fund.models.avalanche_consensus import bench_avalanche_consensus
from quant_fund.models.honey_badger import bench_honey_badger
from quant_fund.models.isis_bcast import bench_isis_bcast
from quant_fund.models.snowball_consensus import bench_snowball_consensus
from quant_fund.models.virtual_synchrony import bench_virtual_synchrony


def test_virtual_synchrony():
    assert bench_virtual_synchrony()["synthetic_virtual_synchrony"] == 1.0


def test_isis_bcast():
    assert bench_isis_bcast()["synthetic_isis_bcast"] == 1.0


def test_atomic_bcast():
    assert bench_atomic_bcast()["synthetic_atomic_bcast"] == 1.0


def test_honey_badger():
    assert bench_honey_badger()["synthetic_honey_badger"] == 1.0


def test_avalanche_consensus():
    assert bench_avalanche_consensus()["synthetic_avalanche_consensus"] == 1.0


def test_snowball_consensus():
    assert bench_snowball_consensus()["synthetic_snowball_consensus"] == 1.0

"""Wave-903 hash-table canon tests."""

from __future__ import annotations

from quant_fund.models.cuckoo_hash import bench_cuckoo_hash
from quant_fund.models.hopscotch_hash import bench_hopscotch_hash
from quant_fund.models.open_addr_hash import bench_open_addr_hash
from quant_fund.models.perfect_hash import bench_perfect_hash
from quant_fund.models.robin_hood_hash import bench_robin_hood_hash
from quant_fund.models.swiss_table import bench_swiss_table


def test_cuckoo_hash():
    assert bench_cuckoo_hash()["synthetic_cuckoo_hash"] == 1.0


def test_hopscotch_hash():
    assert bench_hopscotch_hash()["synthetic_hopscotch_hash"] == 1.0


def test_robin_hood_hash():
    assert bench_robin_hood_hash()["synthetic_robin_hood_hash"] == 1.0


def test_swiss_table():
    assert bench_swiss_table()["synthetic_swiss_table"] == 1.0


def test_open_addr_hash():
    assert bench_open_addr_hash()["synthetic_open_addr_hash"] == 1.0


def test_perfect_hash():
    assert bench_perfect_hash()["synthetic_perfect_hash"] == 1.0

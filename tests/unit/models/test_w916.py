"""Wave-916 data-structures-2 canon tests."""

from __future__ import annotations

from quant_fund.models.da_trie import bench_da_trie
from quant_fund.models.fst_index import bench_fst_index
from quant_fund.models.hollow_dsu import bench_hollow_dsu
from quant_fund.models.hollow_heap import bench_hollow_heap
from quant_fund.models.rank_pairing import bench_rank_pairing
from quant_fund.models.soft_heap import bench_soft_heap


def test_soft_heap():
    assert bench_soft_heap()["synthetic_soft_heap"] == 1.0


def test_hollow_heap():
    assert bench_hollow_heap()["synthetic_hollow_heap"] == 1.0


def test_rank_pairing():
    assert bench_rank_pairing()["synthetic_rank_pairing"] == 1.0


def test_hollow_dsu():
    assert bench_hollow_dsu()["synthetic_hollow_dsu"] == 1.0


def test_da_trie():
    assert bench_da_trie()["synthetic_da_trie"] == 1.0


def test_fst_index():
    assert bench_fst_index()["synthetic_fst_index"] == 1.0

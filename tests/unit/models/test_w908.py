"""Wave-908 range-query canon tests."""

from __future__ import annotations

from quant_fund.models.fenwick_tree import bench_fenwick_tree
from quant_fund.models.merge_sort_tree import bench_merge_sort_tree
from quant_fund.models.segment_tree import bench_segment_tree
from quant_fund.models.sparse_table import bench_sparse_table
from quant_fund.models.sqrt_decomp import bench_sqrt_decomp
from quant_fund.models.wavelet_tree import bench_wavelet_tree


def test_segment_tree():
    assert bench_segment_tree()["synthetic_segment_tree"] == 1.0


def test_fenwick_tree():
    assert bench_fenwick_tree()["synthetic_fenwick_tree"] == 1.0


def test_sparse_table():
    assert bench_sparse_table()["synthetic_sparse_table"] == 1.0


def test_sqrt_decomp():
    assert bench_sqrt_decomp()["synthetic_sqrt_decomp"] == 1.0


def test_wavelet_tree():
    assert bench_wavelet_tree()["synthetic_wavelet_tree"] == 1.0


def test_merge_sort_tree():
    assert bench_merge_sort_tree()["synthetic_merge_sort_tree"] == 1.0

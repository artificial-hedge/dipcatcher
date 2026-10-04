"""Wave-909 deque/linked-structure canon tests."""

from __future__ import annotations

from quant_fund.models.deque_array import bench_deque_array
from quant_fund.models.doubly_linked_list import bench_doubly_linked_list
from quant_fund.models.gap_buffer import bench_gap_buffer
from quant_fund.models.piece_table import bench_piece_table
from quant_fund.models.unrolled_list import bench_unrolled_list
from quant_fund.models.xor_linked_list import bench_xor_linked_list


def test_doubly_linked_list():
    assert bench_doubly_linked_list()["synthetic_doubly_linked_list"] == 1.0


def test_unrolled_list():
    assert bench_unrolled_list()["synthetic_unrolled_list"] == 1.0


def test_gap_buffer():
    assert bench_gap_buffer()["synthetic_gap_buffer"] == 1.0


def test_piece_table():
    assert bench_piece_table()["synthetic_piece_table"] == 1.0


def test_deque_array():
    assert bench_deque_array()["synthetic_deque_array"] == 1.0


def test_xor_linked_list():
    assert bench_xor_linked_list()["synthetic_xor_linked_list"] == 1.0

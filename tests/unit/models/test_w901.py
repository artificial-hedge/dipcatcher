"""Wave-901 heap canon tests."""

from __future__ import annotations

from quant_fund.models.binary_heap import bench_binary_heap
from quant_fund.models.binomial_heap import bench_binomial_heap
from quant_fund.models.fibonacci_heap import bench_fibonacci_heap
from quant_fund.models.leftist_heap import bench_leftist_heap
from quant_fund.models.pairing_heap import bench_pairing_heap
from quant_fund.models.skew_heap import bench_skew_heap


def test_binary_heap():
    assert bench_binary_heap()["synthetic_binary_heap"] == 1.0


def test_fibonacci_heap():
    assert bench_fibonacci_heap()["synthetic_fibonacci_heap"] == 1.0


def test_pairing_heap():
    assert bench_pairing_heap()["synthetic_pairing_heap"] == 1.0


def test_binomial_heap():
    assert bench_binomial_heap()["synthetic_binomial_heap"] == 1.0


def test_leftist_heap():
    assert bench_leftist_heap()["synthetic_leftist_heap"] == 1.0


def test_skew_heap():
    assert bench_skew_heap()["synthetic_skew_heap"] == 1.0

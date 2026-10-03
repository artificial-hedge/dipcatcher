"""Wave-905 sorting canon tests."""

from __future__ import annotations

from quant_fund.models.heapsort import bench_heapsort
from quant_fund.models.introsort import bench_introsort
from quant_fund.models.mergesort import bench_mergesort
from quant_fund.models.quicksort import bench_quicksort
from quant_fund.models.radix_sort import bench_radix_sort
from quant_fund.models.timsort import bench_timsort


def test_quicksort():
    assert bench_quicksort()["synthetic_quicksort"] == 1.0


def test_mergesort():
    assert bench_mergesort()["synthetic_mergesort"] == 1.0


def test_heapsort():
    assert bench_heapsort()["synthetic_heapsort"] == 1.0


def test_introsort():
    assert bench_introsort()["synthetic_introsort"] == 1.0


def test_timsort():
    assert bench_timsort()["synthetic_timsort"] == 1.0


def test_radix_sort():
    assert bench_radix_sort()["synthetic_radix_sort"] == 1.0

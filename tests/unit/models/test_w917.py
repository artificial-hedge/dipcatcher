"""Wave-917 data-structures-3 canon tests."""

from __future__ import annotations

from quant_fund.models.bucket_sort import bench_bucket_sort
from quant_fund.models.chained_hash import bench_chained_hash
from quant_fund.models.linear_probe import bench_linear_probe
from quant_fund.models.rand_access_list import bench_rand_access_list
from quant_fund.models.shell_sort import bench_shell_sort
from quant_fund.models.skew_list import bench_skew_list


def test_chained_hash():
    assert bench_chained_hash()["synthetic_chained_hash"] == 1.0


def test_linear_probe():
    assert bench_linear_probe()["synthetic_linear_probe"] == 1.0


def test_bucket_sort():
    assert bench_bucket_sort()["synthetic_bucket_sort"] == 1.0


def test_shell_sort():
    assert bench_shell_sort()["synthetic_shell_sort"] == 1.0


def test_rand_access_list():
    assert bench_rand_access_list()["synthetic_rand_access_list"] == 1.0


def test_skew_list():
    assert bench_skew_list()["synthetic_skew_list"] == 1.0

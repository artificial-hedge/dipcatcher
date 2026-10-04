"""Wave-906 persistent-structure canon tests."""

from __future__ import annotations

from quant_fund.models.finger_tree import bench_finger_tree
from quant_fund.models.persistent_array import bench_persistent_array
from quant_fund.models.pure_queue import bench_pure_queue
from quant_fund.models.rope_string import bench_rope_string
from quant_fund.models.skip_list import bench_skip_list
from quant_fund.models.vlist import bench_vlist


def test_skip_list():
    assert bench_skip_list()["synthetic_skip_list"] == 1.0


def test_persistent_array():
    assert bench_persistent_array()["synthetic_persistent_array"] == 1.0


def test_finger_tree():
    assert bench_finger_tree()["synthetic_finger_tree"] == 1.0


def test_rope_string():
    assert bench_rope_string()["synthetic_rope_string"] == 1.0


def test_vlist():
    assert bench_vlist()["synthetic_vlist"] == 1.0


def test_pure_queue():
    assert bench_pure_queue()["synthetic_pure_queue"] == 1.0

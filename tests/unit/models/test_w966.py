"""Wave-966 operator-space canon tests."""

from __future__ import annotations

from quant_fund.models.cb_map import bench_cb_map
from quant_fund.models.complete_contraction import bench_complete_contraction
from quant_fund.models.injective_space import bench_injective_space
from quant_fund.models.noncommutative_lp import bench_noncommutative_lp
from quant_fund.models.oh_emb import bench_oh_emb
from quant_fund.models.operator_space import bench_operator_space


def test_operator_space():
    assert bench_operator_space()["synthetic_operator_space"] == 1.0


def test_cb_map():
    assert bench_cb_map()["synthetic_cb_map"] == 1.0


def test_complete_contraction():
    assert bench_complete_contraction()["synthetic_complete_contraction"] == 1.0


def test_injective_space():
    assert bench_injective_space()["synthetic_injective_space"] == 1.0


def test_noncommutative_lp():
    assert bench_noncommutative_lp()["synthetic_noncommutative_lp"] == 1.0


def test_oh_emb():
    assert bench_oh_emb()["synthetic_oh_emb"] == 1.0

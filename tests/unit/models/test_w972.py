"""Wave-972 free-probability-2 canon tests."""

from __future__ import annotations

from quant_fund.models.free_berg import bench_free_berg
from quant_fund.models.free_cumulant import bench_free_cumulant
from quant_fund.models.free_entropy import bench_free_entropy
from quant_fund.models.free_fisher_info import bench_free_fisher_info
from quant_fund.models.freeness_check import bench_freeness_check
from quant_fund.models.matrix_model_free import bench_matrix_model_free


def test_free_entropy():
    assert bench_free_entropy()["synthetic_free_entropy"] == 1.0


def test_free_fisher_info():
    assert bench_free_fisher_info()["synthetic_free_fisher_info"] == 1.0


def test_free_cumulant():
    assert bench_free_cumulant()["synthetic_free_cumulant"] == 1.0


def test_freeness_check():
    assert bench_freeness_check()["synthetic_freeness_check"] == 1.0


def test_matrix_model_free():
    assert bench_matrix_model_free()["synthetic_matrix_model_free"] == 1.0


def test_free_berg():
    assert bench_free_berg()["synthetic_free_berg"] == 1.0

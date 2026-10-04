"""Wave-956 tensor-theory canon tests."""

from __future__ import annotations

from quant_fund.models.cp_rank import bench_cp_rank
from quant_fund.models.mode_n_product import bench_mode_n_product
from quant_fund.models.tensor_norm import bench_tensor_norm
from quant_fund.models.tensor_symmetry import bench_tensor_symmetry
from quant_fund.models.tensor_trace import bench_tensor_trace
from quant_fund.models.tucker_rank import bench_tucker_rank


def test_tucker_rank():
    assert bench_tucker_rank()["synthetic_tucker_rank"] == 1.0


def test_cp_rank():
    assert bench_cp_rank()["synthetic_cp_rank"] == 1.0


def test_tensor_norm():
    assert bench_tensor_norm()["synthetic_tensor_norm"] == 1.0


def test_tensor_trace():
    assert bench_tensor_trace()["synthetic_tensor_trace"] == 1.0


def test_mode_n_product():
    assert bench_mode_n_product()["synthetic_mode_n_product"] == 1.0


def test_tensor_symmetry():
    assert bench_tensor_symmetry()["synthetic_tensor_symmetry"] == 1.0

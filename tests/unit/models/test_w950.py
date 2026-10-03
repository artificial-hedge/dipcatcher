"""Wave-950 tensor-algebra canon tests."""

from __future__ import annotations

from quant_fund.models.hadamard_product import bench_hadamard_product
from quant_fund.models.khatri_rao import bench_khatri_rao
from quant_fund.models.kron_product import bench_kron_product
from quant_fund.models.outer_product import bench_outer_product
from quant_fund.models.tensor_contraction import bench_tensor_contraction
from quant_fund.models.tensor_unfold import bench_tensor_unfold


def test_tensor_contraction():
    assert bench_tensor_contraction()["synthetic_tensor_contraction"] == 1.0


def test_khatri_rao():
    assert bench_khatri_rao()["synthetic_khatri_rao"] == 1.0


def test_kron_product():
    assert bench_kron_product()["synthetic_kron_product"] == 1.0


def test_hadamard_product():
    assert bench_hadamard_product()["synthetic_hadamard_product"] == 1.0


def test_tensor_unfold():
    assert bench_tensor_unfold()["synthetic_tensor_unfold"] == 1.0


def test_outer_product():
    assert bench_outer_product()["synthetic_outer_product"] == 1.0

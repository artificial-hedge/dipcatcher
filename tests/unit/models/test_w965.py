"""Wave-965 Schatten/compact canon tests."""

from __future__ import annotations

from quant_fund.models.compact_normal import bench_compact_normal
from quant_fund.models.hilbert_schmidt_op import bench_hilbert_schmidt_op
from quant_fund.models.polar_operator import bench_polar_operator
from quant_fund.models.schmidt_decomp import bench_schmidt_decomp
from quant_fund.models.singular_value_op import bench_singular_value_op
from quant_fund.models.trace_class_op import bench_trace_class_op


def test_hilbert_schmidt_op():
    assert bench_hilbert_schmidt_op()["synthetic_hilbert_schmidt_op"] == 1.0


def test_trace_class_op():
    assert bench_trace_class_op()["synthetic_trace_class_op"] == 1.0


def test_singular_value_op():
    assert bench_singular_value_op()["synthetic_singular_value_op"] == 1.0


def test_schmidt_decomp():
    assert bench_schmidt_decomp()["synthetic_schmidt_decomp"] == 1.0


def test_compact_normal():
    assert bench_compact_normal()["synthetic_compact_normal"] == 1.0


def test_polar_operator():
    assert bench_polar_operator()["synthetic_polar_operator"] == 1.0

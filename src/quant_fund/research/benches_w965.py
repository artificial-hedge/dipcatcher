"""Wave-965 Schatten/compact canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.compact_normal import bench_compact_normal
from quant_fund.models.hilbert_schmidt_op import bench_hilbert_schmidt_op
from quant_fund.models.polar_operator import bench_polar_operator
from quant_fund.models.schmidt_decomp import bench_schmidt_decomp
from quant_fund.models.singular_value_op import bench_singular_value_op
from quant_fund.models.trace_class_op import bench_trace_class_op

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_hilbert_schmidt_op_family(seed: int = _SEED + 35300) -> dict[str, float]:
    return _finite_blob(bench_hilbert_schmidt_op(seed))


def bench_trace_class_op_family(seed: int = _SEED + 35301) -> dict[str, float]:
    return _finite_blob(bench_trace_class_op(seed))


def bench_singular_value_op_family(seed: int = _SEED + 35302) -> dict[str, float]:
    return _finite_blob(bench_singular_value_op(seed))


def bench_schmidt_decomp_family(seed: int = _SEED + 35303) -> dict[str, float]:
    return _finite_blob(bench_schmidt_decomp(seed))


def bench_compact_normal_family(seed: int = _SEED + 35304) -> dict[str, float]:
    return _finite_blob(bench_compact_normal(seed))


def bench_polar_operator_family(seed: int = _SEED + 35305) -> dict[str, float]:
    return _finite_blob(bench_polar_operator(seed))

"""Wave-957 operator-theory-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fredholm_op import bench_fredholm_op
from quant_fund.models.multiplication_op import bench_multiplication_op
from quant_fund.models.normal_operator import bench_normal_operator
from quant_fund.models.selfadjoint_op import bench_selfadjoint_op
from quant_fund.models.shift_operator import bench_shift_operator
from quant_fund.models.unitary_operator import bench_unitary_operator

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


def bench_selfadjoint_op_family(seed: int = _SEED + 34500) -> dict[str, float]:
    return _finite_blob(bench_selfadjoint_op(seed))


def bench_unitary_operator_family(seed: int = _SEED + 34501) -> dict[str, float]:
    return _finite_blob(bench_unitary_operator(seed))


def bench_shift_operator_family(seed: int = _SEED + 34502) -> dict[str, float]:
    return _finite_blob(bench_shift_operator(seed))


def bench_fredholm_op_family(seed: int = _SEED + 34503) -> dict[str, float]:
    return _finite_blob(bench_fredholm_op(seed))


def bench_normal_operator_family(seed: int = _SEED + 34504) -> dict[str, float]:
    return _finite_blob(bench_normal_operator(seed))


def bench_multiplication_op_family(seed: int = _SEED + 34505) -> dict[str, float]:
    return _finite_blob(bench_multiplication_op(seed))

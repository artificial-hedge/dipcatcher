"""Wave-959 operator-theory-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.accretive_op import bench_accretive_op
from quant_fund.models.contraction_op import bench_contraction_op
from quant_fund.models.differential_op import bench_differential_op
from quant_fund.models.integral_op import bench_integral_op
from quant_fund.models.sectorial_op import bench_sectorial_op
from quant_fund.models.toeplitz_op import bench_toeplitz_op

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


def bench_toeplitz_op_family(seed: int = _SEED + 34700) -> dict[str, float]:
    return _finite_blob(bench_toeplitz_op(seed))


def bench_integral_op_family(seed: int = _SEED + 34701) -> dict[str, float]:
    return _finite_blob(bench_integral_op(seed))


def bench_differential_op_family(seed: int = _SEED + 34702) -> dict[str, float]:
    return _finite_blob(bench_differential_op(seed))


def bench_contraction_op_family(seed: int = _SEED + 34703) -> dict[str, float]:
    return _finite_blob(bench_contraction_op(seed))


def bench_accretive_op_family(seed: int = _SEED + 34704) -> dict[str, float]:
    return _finite_blob(bench_accretive_op(seed))


def bench_sectorial_op_family(seed: int = _SEED + 34705) -> dict[str, float]:
    return _finite_blob(bench_sectorial_op(seed))

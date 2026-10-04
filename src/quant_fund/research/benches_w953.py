"""Wave-953 operator-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bounded_operator import bench_bounded_operator
from quant_fund.models.isometry_operator import bench_isometry_operator
from quant_fund.models.operator_adjoint import bench_operator_adjoint
from quant_fund.models.operator_norm import bench_operator_norm
from quant_fund.models.positive_operator import bench_positive_operator
from quant_fund.models.projection_operator import bench_projection_operator

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


def bench_bounded_operator_family(seed: int = _SEED + 34100) -> dict[str, float]:
    return _finite_blob(bench_bounded_operator(seed))


def bench_operator_norm_family(seed: int = _SEED + 34101) -> dict[str, float]:
    return _finite_blob(bench_operator_norm(seed))


def bench_operator_adjoint_family(seed: int = _SEED + 34102) -> dict[str, float]:
    return _finite_blob(bench_operator_adjoint(seed))


def bench_projection_operator_family(seed: int = _SEED + 34103) -> dict[str, float]:
    return _finite_blob(bench_projection_operator(seed))


def bench_positive_operator_family(seed: int = _SEED + 34104) -> dict[str, float]:
    return _finite_blob(bench_positive_operator(seed))


def bench_isometry_operator_family(seed: int = _SEED + 34105) -> dict[str, float]:
    return _finite_blob(bench_isometry_operator(seed))

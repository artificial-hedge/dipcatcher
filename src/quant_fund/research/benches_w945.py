"""Wave-945 matrix-norm canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hankel_op import bench_hankel_op
from quant_fund.models.kyfan_norm import bench_kyfan_norm
from quant_fund.models.matrix_det import bench_matrix_det
from quant_fund.models.numerical_radius import bench_numerical_radius
from quant_fund.models.pfaffian_poly import bench_pfaffian_poly
from quant_fund.models.schatten_norm import bench_schatten_norm

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


def bench_kyfan_norm_family(seed: int = _SEED + 33300) -> dict[str, float]:
    return _finite_blob(bench_kyfan_norm(seed))


def bench_schatten_norm_family(seed: int = _SEED + 33301) -> dict[str, float]:
    return _finite_blob(bench_schatten_norm(seed))


def bench_numerical_radius_family(seed: int = _SEED + 33302) -> dict[str, float]:
    return _finite_blob(bench_numerical_radius(seed))


def bench_matrix_det_family(seed: int = _SEED + 33303) -> dict[str, float]:
    return _finite_blob(bench_matrix_det(seed))


def bench_pfaffian_poly_family(seed: int = _SEED + 33304) -> dict[str, float]:
    return _finite_blob(bench_pfaffian_poly(seed))


def bench_hankel_op_family(seed: int = _SEED + 33305) -> dict[str, float]:
    return _finite_blob(bench_hankel_op(seed))

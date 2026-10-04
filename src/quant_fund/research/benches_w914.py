"""Wave-914 RK/BVP-methods-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.green_function_bvp import bench_green_function_bvp
from quant_fund.models.invariant_imbedding import bench_invariant_imbedding
from quant_fund.models.ralston_rk import bench_ralston_rk
from quant_fund.models.ralston_second import bench_ralston_second
from quant_fund.models.runge_kutta4 import bench_runge_kutta4
from quant_fund.models.verner_rk import bench_verner_rk

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


def bench_ralston_rk_family(seed: int = _SEED + 30200) -> dict[str, float]:
    return _finite_blob(bench_ralston_rk(seed))


def bench_verner_rk_family(seed: int = _SEED + 30201) -> dict[str, float]:
    return _finite_blob(bench_verner_rk(seed))


def bench_ralston_second_family(seed: int = _SEED + 30202) -> dict[str, float]:
    return _finite_blob(bench_ralston_second(seed))


def bench_runge_kutta4_family(seed: int = _SEED + 30203) -> dict[str, float]:
    return _finite_blob(bench_runge_kutta4(seed))


def bench_invariant_imbedding_family(seed: int = _SEED + 30204) -> dict[str, float]:
    return _finite_blob(bench_invariant_imbedding(seed))


def bench_green_function_bvp_family(seed: int = _SEED + 30205) -> dict[str, float]:
    return _finite_blob(bench_green_function_bvp(seed))

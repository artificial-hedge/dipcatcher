"""Wave-411 algebraic-geometry-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.blow_up import bench_blow_up
from quant_fund.models.divisor_class import bench_divisor_class
from quant_fund.models.dualizing import bench_dualizing
from quant_fund.models.intersection_mult import bench_intersection_mult
from quant_fund.models.normalization import bench_normalization
from quant_fund.models.tangent_cone import bench_tangent_cone

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_blow_up_family(seed: int = _SEED + 2372) -> dict[str, float]:
    return _floats(_finite_blob("blow_up", bench_blow_up(seed)))


def bench_intersection_mult_family(
    seed: int = _SEED + 2373,
) -> dict[str, float]:
    return _floats(_finite_blob("intersection_mult", bench_intersection_mult(seed)))


def bench_tangent_cone_family(
    seed: int = _SEED + 2374,
) -> dict[str, float]:
    return _floats(_finite_blob("tangent_cone", bench_tangent_cone(seed)))


def bench_normalization_family(
    seed: int = _SEED + 2375,
) -> dict[str, float]:
    return _floats(_finite_blob("normalization", bench_normalization(seed)))


def bench_divisor_class_family(
    seed: int = _SEED + 2376,
) -> dict[str, float]:
    return _floats(_finite_blob("divisor_class", bench_divisor_class(seed)))


def bench_dualizing_family(seed: int = _SEED + 2377) -> dict[str, float]:
    return _floats(_finite_blob("dualizing", bench_dualizing(seed)))

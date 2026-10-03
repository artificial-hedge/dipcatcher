"""Wave-678 category-16 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.equipment_cat import bench_equipment_cat
from quant_fund.models.fibrant_cat import bench_fibrant_cat
from quant_fund.models.homotopical_cat import (
    bench_homotopical_cat,
)
from quant_fund.models.pointed_cat import bench_pointed_cat
from quant_fund.models.relative_cat import bench_relative_cat
from quant_fund.models.simplicial_cat import bench_simplicial_cat

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


def bench_simplicial_cat_family(
    seed: int = _SEED + 6700,
) -> dict[str, float]:
    return _floats(_finite_blob("simplicial_cat", bench_simplicial_cat(seed)))


def bench_homotopical_cat_family(
    seed: int = _SEED + 6701,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopical_cat", bench_homotopical_cat(seed)))


def bench_relative_cat_family(
    seed: int = _SEED + 6702,
) -> dict[str, float]:
    return _floats(_finite_blob("relative_cat", bench_relative_cat(seed)))


def bench_equipment_cat_family(
    seed: int = _SEED + 6703,
) -> dict[str, float]:
    return _floats(_finite_blob("equipment_cat", bench_equipment_cat(seed)))


def bench_fibrant_cat_family(
    seed: int = _SEED + 6704,
) -> dict[str, float]:
    return _floats(_finite_blob("fibrant_cat", bench_fibrant_cat(seed)))


def bench_pointed_cat_family(
    seed: int = _SEED + 6705,
) -> dict[str, float]:
    return _floats(_finite_blob("pointed_cat", bench_pointed_cat(seed)))

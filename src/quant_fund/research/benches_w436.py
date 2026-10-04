"""Wave-436 infinity-categories-2 bench adapters (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adjoint_functor import (
    bench_adjoint_functor,
)
from quant_fund.models.bousfield_loc import bench_bousfield_loc
from quant_fund.models.cartesian_fib import bench_cartesian_fib
from quant_fund.models.complete_seg import bench_complete_seg
from quant_fund.models.presentable_cat import bench_presentable_cat
from quant_fund.models.straightening import bench_straightening

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


def bench_complete_seg_family(
    seed: int = _SEED + 2522,
) -> dict[str, float]:
    return _floats(_finite_blob("complete_seg", bench_complete_seg(seed)))


def bench_cartesian_fib_family(
    seed: int = _SEED + 2523,
) -> dict[str, float]:
    return _floats(_finite_blob("cartesian_fib", bench_cartesian_fib(seed)))


def bench_straightening_family(
    seed: int = _SEED + 2524,
) -> dict[str, float]:
    return _floats(_finite_blob("straightening", bench_straightening(seed)))


def bench_presentable_cat_family(
    seed: int = _SEED + 2525,
) -> dict[str, float]:
    return _floats(_finite_blob("presentable_cat", bench_presentable_cat(seed)))


def bench_adjoint_functor_family(
    seed: int = _SEED + 2526,
) -> dict[str, float]:
    return _floats(_finite_blob("adjoint_functor", bench_adjoint_functor(seed)))


def bench_bousfield_loc_family(
    seed: int = _SEED + 2527,
) -> dict[str, float]:
    return _floats(_finite_blob("bousfield_loc", bench_bousfield_loc(seed)))

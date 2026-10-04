"""Wave-696 category-19 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_image import bench_cat_image
from quant_fund.models.cat_index import bench_cat_index
from quant_fund.models.cat_kernel import bench_cat_kernel
from quant_fund.models.cat_monotone import (
    bench_cat_monotone,
)
from quant_fund.models.cat_pullback import (
    bench_cat_pullback,
)
from quant_fund.models.cat_rank import bench_cat_rank

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


def bench_cat_rank_family(
    seed: int = _SEED + 8500,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_rank", bench_cat_rank(seed)))


def bench_cat_index_family(
    seed: int = _SEED + 8501,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_index", bench_cat_index(seed)))


def bench_cat_monotone_family(
    seed: int = _SEED + 8502,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_monotone", bench_cat_monotone(seed)))


def bench_cat_kernel_family(
    seed: int = _SEED + 8503,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_kernel", bench_cat_kernel(seed)))


def bench_cat_image_family(
    seed: int = _SEED + 8504,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_image", bench_cat_image(seed)))


def bench_cat_pullback_family(
    seed: int = _SEED + 8505,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_pullback", bench_cat_pullback(seed)))

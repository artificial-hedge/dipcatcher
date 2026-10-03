"""Wave-469 category-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.accessible_cat import bench_accessible_cat
from quant_fund.models.day_conv import bench_day_conv
from quant_fund.models.derivator2 import bench_derivator2
from quant_fund.models.enriched_cat import bench_enriched_cat
from quant_fund.models.fibered_cat import bench_fibered_cat
from quant_fund.models.weight_lim import bench_weight_lim

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


def bench_enriched_cat_family(seed: int = _SEED + 2720) -> dict[str, float]:
    return _floats(_finite_blob("enriched_cat", bench_enriched_cat(seed)))


def bench_weight_lim_family(seed: int = _SEED + 2721) -> dict[str, float]:
    return _floats(_finite_blob("weight_lim", bench_weight_lim(seed)))


def bench_fibered_cat_family(seed: int = _SEED + 2722) -> dict[str, float]:
    return _floats(_finite_blob("fibered_cat", bench_fibered_cat(seed)))


def bench_derivator2_family(seed: int = _SEED + 2723) -> dict[str, float]:
    return _floats(_finite_blob("derivator2", bench_derivator2(seed)))


def bench_accessible_cat_family(seed: int = _SEED + 2724) -> dict[str, float]:
    return _floats(_finite_blob("accessible_cat", bench_accessible_cat(seed)))


def bench_day_conv_family(seed: int = _SEED + 2725) -> dict[str, float]:
    return _floats(_finite_blob("day_conv", bench_day_conv(seed)))

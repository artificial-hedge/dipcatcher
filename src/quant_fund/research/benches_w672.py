"""Wave-672 category-14 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.compact_cat import bench_compact_cat
from quant_fund.models.monoidal_derived import (
    bench_monoidal_derived,
)
from quant_fund.models.perverse_cat import bench_perverse_cat
from quant_fund.models.smashing_cat import bench_smashing_cat
from quant_fund.models.super_cat import bench_super_cat
from quant_fund.models.tannakian_cat import bench_tannakian_cat

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


def bench_tannakian_cat_family(
    seed: int = _SEED + 6100,
) -> dict[str, float]:
    return _floats(_finite_blob("tannakian_cat", bench_tannakian_cat(seed)))


def bench_super_cat_family(
    seed: int = _SEED + 6101,
) -> dict[str, float]:
    return _floats(_finite_blob("super_cat", bench_super_cat(seed)))


def bench_perverse_cat_family(
    seed: int = _SEED + 6102,
) -> dict[str, float]:
    return _floats(_finite_blob("perverse_cat", bench_perverse_cat(seed)))


def bench_smashing_cat_family(
    seed: int = _SEED + 6103,
) -> dict[str, float]:
    return _floats(_finite_blob("smashing_cat", bench_smashing_cat(seed)))


def bench_compact_cat_family(
    seed: int = _SEED + 6104,
) -> dict[str, float]:
    return _floats(_finite_blob("compact_cat", bench_compact_cat(seed)))


def bench_monoidal_derived_family(
    seed: int = _SEED + 6105,
) -> dict[str, float]:
    return _floats(_finite_blob("monoidal_derived", bench_monoidal_derived(seed)))

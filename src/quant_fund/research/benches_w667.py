"""Wave-667 category-13 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ab_cat import bench_ab_cat
from quant_fund.models.coniveau_fil import bench_coniveau_fil
from quant_fund.models.exact_cat2 import bench_exact_cat2
from quant_fund.models.grothendieck_cat import (
    bench_grothendieck_cat,
)
from quant_fund.models.special_cat import bench_special_cat
from quant_fund.models.stable_cat2 import bench_stable_cat2

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


def bench_stable_cat2_family(
    seed: int = _SEED + 5600,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_cat2", bench_stable_cat2(seed)))


def bench_exact_cat2_family(
    seed: int = _SEED + 5601,
) -> dict[str, float]:
    return _floats(_finite_blob("exact_cat2", bench_exact_cat2(seed)))


def bench_ab_cat_family(
    seed: int = _SEED + 5602,
) -> dict[str, float]:
    return _floats(_finite_blob("ab_cat", bench_ab_cat(seed)))


def bench_grothendieck_cat_family(
    seed: int = _SEED + 5603,
) -> dict[str, float]:
    return _floats(_finite_blob("grothendieck_cat", bench_grothendieck_cat(seed)))


def bench_coniveau_fil_family(
    seed: int = _SEED + 5604,
) -> dict[str, float]:
    return _floats(_finite_blob("coniveau_fil", bench_coniveau_fil(seed)))


def bench_special_cat_family(
    seed: int = _SEED + 5605,
) -> dict[str, float]:
    return _floats(_finite_blob("special_cat", bench_special_cat(seed)))

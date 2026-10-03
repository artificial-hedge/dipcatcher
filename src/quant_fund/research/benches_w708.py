"""Wave-708 category-21 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_dold_kan import bench_cat_dold_kan
from quant_fund.models.cat_enriched_lim import (
    bench_cat_enriched_lim,
)
from quant_fund.models.cat_hoc import bench_cat_hoc
from quant_fund.models.cat_pseudo_limit import (
    bench_cat_pseudo_limit,
)
from quant_fund.models.cat_reedy_cat import bench_cat_reedy_cat
from quant_fund.models.cat_weak_eq import bench_cat_weak_eq

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


def bench_cat_pseudo_limit_family(
    seed: int = _SEED + 9700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cat_pseudo_limit",
            bench_cat_pseudo_limit(seed),
        )
    )


def bench_cat_weak_eq_family(
    seed: int = _SEED + 9701,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_weak_eq", bench_cat_weak_eq(seed)))


def bench_cat_reedy_cat_family(
    seed: int = _SEED + 9702,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_reedy_cat", bench_cat_reedy_cat(seed)))


def bench_cat_dold_kan_family(
    seed: int = _SEED + 9703,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_dold_kan", bench_cat_dold_kan(seed)))


def bench_cat_hoc_family(
    seed: int = _SEED + 9704,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_hoc", bench_cat_hoc(seed)))


def bench_cat_enriched_lim_family(
    seed: int = _SEED + 9705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cat_enriched_lim",
            bench_cat_enriched_lim(seed),
        )
    )

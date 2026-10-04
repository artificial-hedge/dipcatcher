"""Wave-677 category-15 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_dg import bench_cat_dg
from quant_fund.models.cat_structure import bench_cat_structure
from quant_fund.models.combinatorial_mc import (
    bench_combinatorial_mc,
)
from quant_fund.models.derivator_cat import bench_derivator_cat
from quant_fund.models.quillen_cat import bench_quillen_cat
from quant_fund.models.univalent_cat import bench_univalent_cat

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


def bench_derivator_cat_family(
    seed: int = _SEED + 6600,
) -> dict[str, float]:
    return _floats(_finite_blob("derivator_cat", bench_derivator_cat(seed)))


def bench_quillen_cat_family(
    seed: int = _SEED + 6601,
) -> dict[str, float]:
    return _floats(_finite_blob("quillen_cat", bench_quillen_cat(seed)))


def bench_combinatorial_mc_family(
    seed: int = _SEED + 6602,
) -> dict[str, float]:
    return _floats(_finite_blob("combinatorial_mc", bench_combinatorial_mc(seed)))


def bench_cat_dg_family(
    seed: int = _SEED + 6603,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_dg", bench_cat_dg(seed)))


def bench_univalent_cat_family(
    seed: int = _SEED + 6604,
) -> dict[str, float]:
    return _floats(_finite_blob("univalent_cat", bench_univalent_cat(seed)))


def bench_cat_structure_family(
    seed: int = _SEED + 6605,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_structure", bench_cat_structure(seed)))

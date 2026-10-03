"""Wave-325 PL-8 ownership/substructural canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borrow_check import bench_borrow_check
from quant_fund.models.capability_perm import bench_capability_perm
from quant_fund.models.escape_region import bench_escape_region
from quant_fund.models.lifetime_outlives import bench_lifetime_outlives
from quant_fund.models.linear_use import bench_linear_use
from quant_fund.models.refinement_liquid import bench_refinement_liquid

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


def bench_borrow_check_family(seed: int = _SEED + 1857) -> dict[str, float]:
    return _floats(_finite_blob("borrow_check", bench_borrow_check(seed)))


def bench_lifetime_outlives_family(seed: int = _SEED + 1858) -> dict[str, float]:
    return _floats(_finite_blob("lifetime_outlives", bench_lifetime_outlives(seed)))


def bench_linear_use_family(seed: int = _SEED + 1859) -> dict[str, float]:
    return _floats(_finite_blob("linear_use", bench_linear_use(seed)))


def bench_escape_region_family(seed: int = _SEED + 1860) -> dict[str, float]:
    return _floats(_finite_blob("escape_region", bench_escape_region(seed)))


def bench_capability_perm_family(seed: int = _SEED + 1861) -> dict[str, float]:
    return _floats(_finite_blob("capability_perm", bench_capability_perm(seed)))


def bench_refinement_liquid_family(seed: int = _SEED + 1862) -> dict[str, float]:
    return _floats(_finite_blob("refinement_liquid", bench_refinement_liquid(seed)))

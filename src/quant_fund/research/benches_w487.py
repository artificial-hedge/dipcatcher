"""Wave-487 category-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.comma_cat import bench_comma_cat
from quant_fund.models.compact_obj import bench_compact_obj
from quant_fund.models.dualizable_cat import bench_dualizable_cat
from quant_fund.models.endo_prof import bench_endo_prof
from quant_fund.models.exact_cat import bench_exact_cat
from quant_fund.models.prestack import bench_prestack

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


def bench_compact_obj_family(seed: int = _SEED + 2828) -> dict[str, float]:
    return _floats(_finite_blob("compact_obj", bench_compact_obj(seed)))


def bench_dualizable_cat_family(seed: int = _SEED + 2829) -> dict[str, float]:
    return _floats(_finite_blob("dualizable_cat", bench_dualizable_cat(seed)))


def bench_comma_cat_family(seed: int = _SEED + 2830) -> dict[str, float]:
    return _floats(_finite_blob("comma_cat", bench_comma_cat(seed)))


def bench_prestack_family(seed: int = _SEED + 2831) -> dict[str, float]:
    return _floats(_finite_blob("prestack", bench_prestack(seed)))


def bench_endo_prof_family(seed: int = _SEED + 2832) -> dict[str, float]:
    return _floats(_finite_blob("endo_prof", bench_endo_prof(seed)))


def bench_exact_cat_family(seed: int = _SEED + 2833) -> dict[str, float]:
    return _floats(_finite_blob("exact_cat", bench_exact_cat(seed)))

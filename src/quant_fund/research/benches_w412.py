"""Wave-412 2-category bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bicat_comp import bench_bicat_comp
from quant_fund.models.cat_enriched import bench_cat_enriched
from quant_fund.models.double_cat import bench_double_cat
from quant_fund.models.lax_functor import bench_lax_functor
from quant_fund.models.mate_calc import bench_mate_calc
from quant_fund.models.two_cat import bench_two_cat

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


def bench_two_cat_family(seed: int = _SEED + 2378) -> dict[str, float]:
    return _floats(_finite_blob("two_cat", bench_two_cat(seed)))


def bench_bicat_comp_family(seed: int = _SEED + 2379) -> dict[str, float]:
    return _floats(_finite_blob("bicat_comp", bench_bicat_comp(seed)))


def bench_mate_calc_family(seed: int = _SEED + 2380) -> dict[str, float]:
    return _floats(_finite_blob("mate_calc", bench_mate_calc(seed)))


def bench_double_cat_family(seed: int = _SEED + 2381) -> dict[str, float]:
    return _floats(_finite_blob("double_cat", bench_double_cat(seed)))


def bench_lax_functor_family(
    seed: int = _SEED + 2382,
) -> dict[str, float]:
    return _floats(_finite_blob("lax_functor", bench_lax_functor(seed)))


def bench_cat_enriched_family(
    seed: int = _SEED + 2383,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_enriched", bench_cat_enriched(seed)))

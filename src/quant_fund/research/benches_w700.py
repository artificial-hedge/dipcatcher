"""Wave-700 category-20 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_lax import bench_cat_lax
from quant_fund.models.cat_pushout import bench_cat_pushout
from quant_fund.models.cat_size import bench_cat_size
from quant_fund.models.cat_span import bench_cat_span
from quant_fund.models.cat_street import bench_cat_street
from quant_fund.models.cat_total import bench_cat_total

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


def bench_cat_pushout_family(
    seed: int = _SEED + 8900,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_pushout", bench_cat_pushout(seed)))


def bench_cat_span_family(
    seed: int = _SEED + 8901,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_span", bench_cat_span(seed)))


def bench_cat_lax_family(
    seed: int = _SEED + 8902,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_lax", bench_cat_lax(seed)))


def bench_cat_street_family(
    seed: int = _SEED + 8903,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_street", bench_cat_street(seed)))


def bench_cat_size_family(
    seed: int = _SEED + 8904,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_size", bench_cat_size(seed)))


def bench_cat_total_family(
    seed: int = _SEED + 8905,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_total", bench_cat_total(seed)))

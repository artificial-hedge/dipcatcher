"""Wave-687 category-17 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_bicomplete import bench_cat_bicomplete
from quant_fund.models.cat_cofibrant import bench_cat_cofibrant
from quant_fund.models.cat_descent import bench_cat_descent
from quant_fund.models.cat_fibrant_obj import (
    bench_cat_fibrant_obj,
)
from quant_fund.models.cat_glueable import bench_cat_glueable
from quant_fund.models.cat_univariant import bench_cat_univariant

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


def bench_cat_fibrant_obj_family(
    seed: int = _SEED + 7600,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_fibrant_obj", bench_cat_fibrant_obj(seed)))


def bench_cat_cofibrant_family(
    seed: int = _SEED + 7601,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_cofibrant", bench_cat_cofibrant(seed)))


def bench_cat_bicomplete_family(
    seed: int = _SEED + 7602,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_bicomplete", bench_cat_bicomplete(seed)))


def bench_cat_univariant_family(
    seed: int = _SEED + 7603,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_univariant", bench_cat_univariant(seed)))


def bench_cat_descent_family(
    seed: int = _SEED + 7604,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_descent", bench_cat_descent(seed)))


def bench_cat_glueable_family(
    seed: int = _SEED + 7605,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_glueable", bench_cat_glueable(seed)))

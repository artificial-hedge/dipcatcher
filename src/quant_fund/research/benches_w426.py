"""Wave-426 homotopy-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_alg import bench_derived_alg
from quant_fund.models.infinity_cat import bench_infinity_cat
from quant_fund.models.model_category import bench_model_category
from quant_fund.models.quillen_adj import bench_quillen_adj
from quant_fund.models.simplicial_set import bench_simplicial_set
from quant_fund.models.stable_cat import bench_stable_cat

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


def bench_model_category_family(
    seed: int = _SEED + 2462,
) -> dict[str, float]:
    return _floats(_finite_blob("model_category", bench_model_category(seed)))


def bench_quillen_adj_family(
    seed: int = _SEED + 2463,
) -> dict[str, float]:
    return _floats(_finite_blob("quillen_adj", bench_quillen_adj(seed)))


def bench_simplicial_set_family(
    seed: int = _SEED + 2464,
) -> dict[str, float]:
    return _floats(_finite_blob("simplicial_set", bench_simplicial_set(seed)))


def bench_infinity_cat_family(
    seed: int = _SEED + 2465,
) -> dict[str, float]:
    return _floats(_finite_blob("infinity_cat", bench_infinity_cat(seed)))


def bench_derived_alg_family(
    seed: int = _SEED + 2466,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_alg", bench_derived_alg(seed)))


def bench_stable_cat_family(
    seed: int = _SEED + 2467,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_cat", bench_stable_cat(seed)))

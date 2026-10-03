"""Wave-391 category-theory-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.closed_cat import bench_closed_cat
from quant_fund.models.distributor import bench_distributor
from quant_fund.models.equivalence_cat import bench_equivalence_cat
from quant_fund.models.kan_extension import bench_kan_extension
from quant_fund.models.monoidal_cat import bench_monoidal_cat
from quant_fund.models.presheaf import bench_presheaf

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


def bench_monoidal_cat_family(seed: int = _SEED + 2252) -> dict[str, float]:
    return _floats(_finite_blob("monoidal_cat", bench_monoidal_cat(seed)))


def bench_closed_cat_family(seed: int = _SEED + 2253) -> dict[str, float]:
    return _floats(_finite_blob("closed_cat", bench_closed_cat(seed)))


def bench_presheaf_family(seed: int = _SEED + 2254) -> dict[str, float]:
    return _floats(_finite_blob("presheaf", bench_presheaf(seed)))


def bench_kan_extension_family(seed: int = _SEED + 2255) -> dict[str, float]:
    return _floats(_finite_blob("kan_extension", bench_kan_extension(seed)))


def bench_distributor_family(seed: int = _SEED + 2256) -> dict[str, float]:
    return _floats(_finite_blob("distributor", bench_distributor(seed)))


def bench_equivalence_cat_family(seed: int = _SEED + 2257) -> dict[str, float]:
    return _floats(_finite_blob("equivalence_cat", bench_equivalence_cat(seed)))

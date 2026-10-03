"""Wave-379 matroid-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dual_matroid import bench_dual_matroid
from quant_fund.models.greedy_matroid import bench_greedy_matroid
from quant_fund.models.matroid_axioms import bench_matroid_axioms
from quant_fund.models.matroid_intersect import bench_matroid_intersect
from quant_fund.models.matroid_union import bench_matroid_union
from quant_fund.models.represented_matroid import bench_represented_matroid

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


def bench_matroid_axioms_family(seed: int = _SEED + 2180) -> dict[str, float]:
    return _floats(_finite_blob("matroid_axioms", bench_matroid_axioms(seed)))


def bench_greedy_matroid_family(seed: int = _SEED + 2181) -> dict[str, float]:
    return _floats(_finite_blob("greedy_matroid", bench_greedy_matroid(seed)))


def bench_matroid_intersect_family(seed: int = _SEED + 2182) -> dict[str, float]:
    return _floats(_finite_blob("matroid_intersect", bench_matroid_intersect(seed)))


def bench_dual_matroid_family(seed: int = _SEED + 2183) -> dict[str, float]:
    return _floats(_finite_blob("dual_matroid", bench_dual_matroid(seed)))


def bench_matroid_union_family(seed: int = _SEED + 2184) -> dict[str, float]:
    return _floats(_finite_blob("matroid_union", bench_matroid_union(seed)))


def bench_represented_matroid_family(seed: int = _SEED + 2185) -> dict[str, float]:
    return _floats(_finite_blob("represented_matroid", bench_represented_matroid(seed)))

"""Wave-348 graph-theory/combinatorics-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.euler_trail import bench_euler_trail
from quant_fund.models.graph_coloring import bench_graph_coloring
from quant_fund.models.matroid_greedy import bench_matroid_greedy
from quant_fund.models.planar_check import bench_planar_check
from quant_fund.models.poset_dimension import bench_poset_dimension
from quant_fund.models.ramsey_r33 import bench_ramsey_r33

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


def bench_graph_coloring_family(seed: int = _SEED + 1995) -> dict[str, float]:
    return _floats(_finite_blob("graph_coloring", bench_graph_coloring(seed)))


def bench_euler_trail_family(seed: int = _SEED + 1996) -> dict[str, float]:
    return _floats(_finite_blob("euler_trail", bench_euler_trail(seed)))


def bench_matroid_greedy_family(seed: int = _SEED + 1997) -> dict[str, float]:
    return _floats(_finite_blob("matroid_greedy", bench_matroid_greedy(seed)))


def bench_planar_check_family(seed: int = _SEED + 1998) -> dict[str, float]:
    return _floats(_finite_blob("planar_check", bench_planar_check(seed)))


def bench_poset_dimension_family(seed: int = _SEED + 1999) -> dict[str, float]:
    return _floats(_finite_blob("poset_dimension", bench_poset_dimension(seed)))


def bench_ramsey_r33_family(seed: int = _SEED + 2000) -> dict[str, float]:
    return _floats(_finite_blob("ramsey_r33", bench_ramsey_r33(seed)))

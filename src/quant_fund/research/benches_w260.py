"""Wave-260 adapters: matching canon — Gale-Shapley, Hopcroft-
Karp, Hungarian, König cover, Gale-Chu quotas, Kahn
layers — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gale_chu import bench_gale_chu
from quant_fund.models.gale_shapley import bench_gale_shapley
from quant_fund.models.hopcroft_karp import bench_hopcroft_karp
from quant_fund.models.konig_cover import bench_konig_cover
from quant_fund.models.kuhn_munkres import bench_kuhn_munkres
from quant_fund.models.topo_layers import bench_topo_layers

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_gale_shapley_family(seed: int = _SEED + 1370) -> dict[str, float]:
    return bench_gale_shapley(seed)


def bench_hopcroft_karp_family(seed: int = _SEED + 1371) -> dict[str, float]:
    return bench_hopcroft_karp(seed)


def bench_kuhn_munkres_family(seed: int = _SEED + 1372) -> dict[str, float]:
    return bench_kuhn_munkres(seed)


def bench_konig_cover_family(seed: int = _SEED + 1373) -> dict[str, float]:
    return bench_konig_cover(seed)


def bench_gale_chu_family(seed: int = _SEED + 1374) -> dict[str, float]:
    return bench_gale_chu(seed)


def bench_topo_layers_family(seed: int = _SEED + 1375) -> dict[str, float]:
    return bench_topo_layers(seed)

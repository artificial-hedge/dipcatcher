"""Wave-250 adapters: graph-3 canon — Dinic max-flow, min-cost
flow (SSP+potentials), A*, bidirectional Dijkstra, CPM critical
path, Bron–Kerbosch cliques — SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.astar_search import bench_astar_search
from quant_fund.models.bidirectional_dijkstra import bench_bidirectional_dijkstra
from quant_fund.models.bron_kerbosch import bench_bron_kerbosch
from quant_fund.models.critical_path import bench_critical_path
from quant_fund.models.dinic_flow import bench_dinic_flow
from quant_fund.models.mincost_flow import bench_mincost_flow

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


def bench_astar_search_family(seed: int = _SEED + 1270) -> dict[str, float]:
    return bench_astar_search(seed)


def bench_bidirectional_dijkstra_family(seed: int = _SEED + 1271) -> dict[str, float]:
    return bench_bidirectional_dijkstra(seed)


def bench_bron_kerbosch_family(seed: int = _SEED + 1272) -> dict[str, float]:
    return bench_bron_kerbosch(seed)


def bench_critical_path_family(seed: int = _SEED + 1273) -> dict[str, float]:
    return bench_critical_path(seed)


def bench_dinic_flow_family(seed: int = _SEED + 1274) -> dict[str, float]:
    return bench_dinic_flow(seed)


def bench_mincost_flow_family(seed: int = _SEED + 1275) -> dict[str, float]:
    return bench_mincost_flow(seed)

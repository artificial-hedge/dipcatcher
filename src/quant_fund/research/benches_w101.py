"""Wave-101 adapters: combinatorial canon — BFS/DFS/Kahn
topological sort + bipartite coloring, Dijkstra/Bellman-Ford/
A*/Floyd-Warshall shortest paths, Dinic max-flow with min-cut
duality, Hungarian + Hopcroft-Karp assignment, Tarjan SCC +
bridges/articulation decomposition, and Knuth Algorithm X
exact cover.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.assignment import bench_assignment
from quant_fund.models.exact_cover import bench_exact_cover
from quant_fund.models.graph_components import bench_graph_components
from quant_fund.models.graph_traversal import bench_graph_traversal
from quant_fund.models.network_flow import bench_network_flow
from quant_fund.models.shortest_paths import bench_shortest_paths

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
                flat[f"{k}_{i}"] = f
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_graph_traversal_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("graph_traversal", bench_graph_traversal(seed=_SEED + 594))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"graph_traversal bench failed: {exc}") from exc


def bench_shortest_paths_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("shortest_paths", bench_shortest_paths(seed=_SEED + 595))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"shortest_paths bench failed: {exc}") from exc


def bench_network_flow_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("network_flow", bench_network_flow(seed=_SEED + 596))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"network_flow bench failed: {exc}") from exc


def bench_assignment_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("assignment", bench_assignment(seed=_SEED + 597)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"assignment bench failed: {exc}") from exc


def bench_graph_components_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("graph_components", bench_graph_components(seed=_SEED + 598))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"graph_components bench failed: {exc}") from exc


def bench_exact_cover_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("exact_cover", bench_exact_cover(seed=_SEED + 599)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"exact_cover bench failed: {exc}") from exc

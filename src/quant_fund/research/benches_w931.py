"""Wave-931 population-metaheuristics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ant_colony import bench_ant_colony
from quant_fund.models.diff_evolution import bench_diff_evolution
from quant_fund.models.firefly_algo import bench_firefly_algo
from quant_fund.models.genetic_tsp import bench_genetic_tsp
from quant_fund.models.harmony_search import bench_harmony_search
from quant_fund.models.pso_swarm import bench_pso_swarm

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_ant_colony_family(seed: int = _SEED + 31900) -> dict[str, float]:
    return _finite_blob(bench_ant_colony(seed))


def bench_pso_swarm_family(seed: int = _SEED + 31901) -> dict[str, float]:
    return _finite_blob(bench_pso_swarm(seed))


def bench_diff_evolution_family(seed: int = _SEED + 31902) -> dict[str, float]:
    return _finite_blob(bench_diff_evolution(seed))


def bench_genetic_tsp_family(seed: int = _SEED + 31903) -> dict[str, float]:
    return _finite_blob(bench_genetic_tsp(seed))


def bench_firefly_algo_family(seed: int = _SEED + 31904) -> dict[str, float]:
    return _finite_blob(bench_firefly_algo(seed))


def bench_harmony_search_family(seed: int = _SEED + 31905) -> dict[str, float]:
    return _finite_blob(bench_harmony_search(seed))

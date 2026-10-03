"""Wave-932 local-search-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.guided_local import bench_guided_local
from quant_fund.models.large_neighborhood import bench_large_neighborhood
from quant_fund.models.path_relinking import bench_path_relinking
from quant_fund.models.ruin_recreate import bench_ruin_recreate
from quant_fund.models.simulated_annealing import bench_simulated_annealing
from quant_fund.models.vns_search import bench_vns_search

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


def bench_vns_search_family(seed: int = _SEED + 32000) -> dict[str, float]:
    return _finite_blob(bench_vns_search(seed))


def bench_large_neighborhood_family(seed: int = _SEED + 32001) -> dict[str, float]:
    return _finite_blob(bench_large_neighborhood(seed))


def bench_ruin_recreate_family(seed: int = _SEED + 32002) -> dict[str, float]:
    return _finite_blob(bench_ruin_recreate(seed))


def bench_path_relinking_family(seed: int = _SEED + 32003) -> dict[str, float]:
    return _finite_blob(bench_path_relinking(seed))


def bench_guided_local_family(seed: int = _SEED + 32004) -> dict[str, float]:
    return _finite_blob(bench_guided_local(seed))


def bench_simulated_annealing_family(seed: int = _SEED + 32005) -> dict[str, float]:
    return _finite_blob(bench_simulated_annealing(seed))

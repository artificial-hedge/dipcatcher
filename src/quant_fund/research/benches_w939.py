"""Wave-939 set-feasibility canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cq_algorithm import bench_cq_algorithm
from quant_fund.models.dykstra_proj import bench_dykstra_proj
from quant_fund.models.halpern_iter import bench_halpern_iter
from quant_fund.models.haugazeau_proj import bench_haugazeau_proj
from quant_fund.models.parallel_prox import bench_parallel_prox
from quant_fund.models.split_feasibility import bench_split_feasibility

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


def bench_split_feasibility_family(seed: int = _SEED + 32700) -> dict[str, float]:
    return _finite_blob(bench_split_feasibility(seed))


def bench_cq_algorithm_family(seed: int = _SEED + 32701) -> dict[str, float]:
    return _finite_blob(bench_cq_algorithm(seed))


def bench_dykstra_proj_family(seed: int = _SEED + 32702) -> dict[str, float]:
    return _finite_blob(bench_dykstra_proj(seed))


def bench_haugazeau_proj_family(seed: int = _SEED + 32703) -> dict[str, float]:
    return _finite_blob(bench_haugazeau_proj(seed))


def bench_parallel_prox_family(seed: int = _SEED + 32704) -> dict[str, float]:
    return _finite_blob(bench_parallel_prox(seed))


def bench_halpern_iter_family(seed: int = _SEED + 32705) -> dict[str, float]:
    return _finite_blob(bench_halpern_iter(seed))

"""Wave-930 metaheuristics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.grasp_meta import bench_grasp_meta
from quant_fund.models.iterated_local import bench_iterated_local
from quant_fund.models.lin_kernighan import bench_lin_kernighan
from quant_fund.models.tabu_search import bench_tabu_search
from quant_fund.models.three_opt_move import bench_three_opt_move
from quant_fund.models.two_opt_move import bench_two_opt_move

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


def bench_lin_kernighan_family(seed: int = _SEED + 31800) -> dict[str, float]:
    return _finite_blob(bench_lin_kernighan(seed))


def bench_two_opt_move_family(seed: int = _SEED + 31801) -> dict[str, float]:
    return _finite_blob(bench_two_opt_move(seed))


def bench_three_opt_move_family(seed: int = _SEED + 31802) -> dict[str, float]:
    return _finite_blob(bench_three_opt_move(seed))


def bench_tabu_search_family(seed: int = _SEED + 31803) -> dict[str, float]:
    return _finite_blob(bench_tabu_search(seed))


def bench_iterated_local_family(seed: int = _SEED + 31804) -> dict[str, float]:
    return _finite_blob(bench_iterated_local(seed))


def bench_grasp_meta_family(seed: int = _SEED + 31805) -> dict[str, float]:
    return _finite_blob(bench_grasp_meta(seed))

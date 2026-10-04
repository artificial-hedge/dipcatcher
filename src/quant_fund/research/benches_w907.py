"""Wave-907 union-find + priority-queue canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dsu_rollback import bench_dsu_rollback
from quant_fund.models.interval_heap import bench_interval_heap
from quant_fund.models.potential_dsu import bench_potential_dsu
from quant_fund.models.union_find import bench_union_find
from quant_fund.models.van_emde_boas import bench_van_emde_boas
from quant_fund.models.weak_heap import bench_weak_heap

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


def bench_union_find_family(seed: int = _SEED + 29500) -> dict[str, float]:
    return _finite_blob(bench_union_find(seed))


def bench_dsu_rollback_family(seed: int = _SEED + 29501) -> dict[str, float]:
    return _finite_blob(bench_dsu_rollback(seed))


def bench_potential_dsu_family(seed: int = _SEED + 29502) -> dict[str, float]:
    return _finite_blob(bench_potential_dsu(seed))


def bench_van_emde_boas_family(seed: int = _SEED + 29503) -> dict[str, float]:
    return _finite_blob(bench_van_emde_boas(seed))


def bench_interval_heap_family(seed: int = _SEED + 29504) -> dict[str, float]:
    return _finite_blob(bench_interval_heap(seed))


def bench_weak_heap_family(seed: int = _SEED + 29505) -> dict[str, float]:
    return _finite_blob(bench_weak_heap(seed))

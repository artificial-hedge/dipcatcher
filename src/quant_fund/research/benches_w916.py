"""Wave-916 data-structures-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.da_trie import bench_da_trie
from quant_fund.models.fst_index import bench_fst_index
from quant_fund.models.hollow_dsu import bench_hollow_dsu
from quant_fund.models.hollow_heap import bench_hollow_heap
from quant_fund.models.rank_pairing import bench_rank_pairing
from quant_fund.models.soft_heap import bench_soft_heap

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


def bench_soft_heap_family(seed: int = _SEED + 30400) -> dict[str, float]:
    return _finite_blob(bench_soft_heap(seed))


def bench_hollow_heap_family(seed: int = _SEED + 30401) -> dict[str, float]:
    return _finite_blob(bench_hollow_heap(seed))


def bench_rank_pairing_family(seed: int = _SEED + 30402) -> dict[str, float]:
    return _finite_blob(bench_rank_pairing(seed))


def bench_hollow_dsu_family(seed: int = _SEED + 30403) -> dict[str, float]:
    return _finite_blob(bench_hollow_dsu(seed))


def bench_da_trie_family(seed: int = _SEED + 30404) -> dict[str, float]:
    return _finite_blob(bench_da_trie(seed))


def bench_fst_index_family(seed: int = _SEED + 30405) -> dict[str, float]:
    return _finite_blob(bench_fst_index(seed))

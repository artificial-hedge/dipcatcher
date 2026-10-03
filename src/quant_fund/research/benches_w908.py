"""Wave-908 range-query canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fenwick_tree import bench_fenwick_tree
from quant_fund.models.merge_sort_tree import bench_merge_sort_tree
from quant_fund.models.segment_tree import bench_segment_tree
from quant_fund.models.sparse_table import bench_sparse_table
from quant_fund.models.sqrt_decomp import bench_sqrt_decomp
from quant_fund.models.wavelet_tree import bench_wavelet_tree

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


def bench_segment_tree_family(seed: int = _SEED + 29600) -> dict[str, float]:
    return _finite_blob(bench_segment_tree(seed))


def bench_fenwick_tree_family(seed: int = _SEED + 29601) -> dict[str, float]:
    return _finite_blob(bench_fenwick_tree(seed))


def bench_sparse_table_family(seed: int = _SEED + 29602) -> dict[str, float]:
    return _finite_blob(bench_sparse_table(seed))


def bench_sqrt_decomp_family(seed: int = _SEED + 29603) -> dict[str, float]:
    return _finite_blob(bench_sqrt_decomp(seed))


def bench_wavelet_tree_family(seed: int = _SEED + 29604) -> dict[str, float]:
    return _finite_blob(bench_wavelet_tree(seed))


def bench_merge_sort_tree_family(seed: int = _SEED + 29605) -> dict[str, float]:
    return _finite_blob(bench_merge_sort_tree(seed))

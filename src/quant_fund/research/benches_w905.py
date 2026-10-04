"""Wave-905 sorting canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.heapsort import bench_heapsort
from quant_fund.models.introsort import bench_introsort
from quant_fund.models.mergesort import bench_mergesort
from quant_fund.models.quicksort import bench_quicksort
from quant_fund.models.radix_sort import bench_radix_sort
from quant_fund.models.timsort import bench_timsort

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


def bench_quicksort_family(seed: int = _SEED + 29300) -> dict[str, float]:
    return _finite_blob(bench_quicksort(seed))


def bench_mergesort_family(seed: int = _SEED + 29301) -> dict[str, float]:
    return _finite_blob(bench_mergesort(seed))


def bench_heapsort_family(seed: int = _SEED + 29302) -> dict[str, float]:
    return _finite_blob(bench_heapsort(seed))


def bench_introsort_family(seed: int = _SEED + 29303) -> dict[str, float]:
    return _finite_blob(bench_introsort(seed))


def bench_timsort_family(seed: int = _SEED + 29304) -> dict[str, float]:
    return _finite_blob(bench_timsort(seed))


def bench_radix_sort_family(seed: int = _SEED + 29305) -> dict[str, float]:
    return _finite_blob(bench_radix_sort(seed))

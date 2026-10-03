"""Wave-901 heap canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.binary_heap import bench_binary_heap
from quant_fund.models.binomial_heap import bench_binomial_heap
from quant_fund.models.fibonacci_heap import bench_fibonacci_heap
from quant_fund.models.leftist_heap import bench_leftist_heap
from quant_fund.models.pairing_heap import bench_pairing_heap
from quant_fund.models.skew_heap import bench_skew_heap

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


def bench_binary_heap_family(seed: int = _SEED + 28900) -> dict[str, float]:
    return _finite_blob(bench_binary_heap(seed))


def bench_fibonacci_heap_family(seed: int = _SEED + 28901) -> dict[str, float]:
    return _finite_blob(bench_fibonacci_heap(seed))


def bench_pairing_heap_family(seed: int = _SEED + 28902) -> dict[str, float]:
    return _finite_blob(bench_pairing_heap(seed))


def bench_binomial_heap_family(seed: int = _SEED + 28903) -> dict[str, float]:
    return _finite_blob(bench_binomial_heap(seed))


def bench_leftist_heap_family(seed: int = _SEED + 28904) -> dict[str, float]:
    return _finite_blob(bench_leftist_heap(seed))


def bench_skew_heap_family(seed: int = _SEED + 28905) -> dict[str, float]:
    return _finite_blob(bench_skew_heap(seed))

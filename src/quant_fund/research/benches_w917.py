"""Wave-917 data-structures-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bucket_sort import bench_bucket_sort
from quant_fund.models.chained_hash import bench_chained_hash
from quant_fund.models.linear_probe import bench_linear_probe
from quant_fund.models.rand_access_list import bench_rand_access_list
from quant_fund.models.shell_sort import bench_shell_sort
from quant_fund.models.skew_list import bench_skew_list

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


def bench_chained_hash_family(seed: int = _SEED + 30500) -> dict[str, float]:
    return _finite_blob(bench_chained_hash(seed))


def bench_linear_probe_family(seed: int = _SEED + 30501) -> dict[str, float]:
    return _finite_blob(bench_linear_probe(seed))


def bench_bucket_sort_family(seed: int = _SEED + 30502) -> dict[str, float]:
    return _finite_blob(bench_bucket_sort(seed))


def bench_shell_sort_family(seed: int = _SEED + 30503) -> dict[str, float]:
    return _finite_blob(bench_shell_sort(seed))


def bench_rand_access_list_family(seed: int = _SEED + 30504) -> dict[str, float]:
    return _finite_blob(bench_rand_access_list(seed))


def bench_skew_list_family(seed: int = _SEED + 30505) -> dict[str, float]:
    return _finite_blob(bench_skew_list(seed))

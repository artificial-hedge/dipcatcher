"""Wave-903 hash-table canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cuckoo_hash import bench_cuckoo_hash
from quant_fund.models.hopscotch_hash import bench_hopscotch_hash
from quant_fund.models.open_addr_hash import bench_open_addr_hash
from quant_fund.models.perfect_hash import bench_perfect_hash
from quant_fund.models.robin_hood_hash import bench_robin_hood_hash
from quant_fund.models.swiss_table import bench_swiss_table

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


def bench_cuckoo_hash_family(seed: int = _SEED + 29100) -> dict[str, float]:
    return _finite_blob(bench_cuckoo_hash(seed))


def bench_hopscotch_hash_family(seed: int = _SEED + 29101) -> dict[str, float]:
    return _finite_blob(bench_hopscotch_hash(seed))


def bench_robin_hood_hash_family(seed: int = _SEED + 29102) -> dict[str, float]:
    return _finite_blob(bench_robin_hood_hash(seed))


def bench_swiss_table_family(seed: int = _SEED + 29103) -> dict[str, float]:
    return _finite_blob(bench_swiss_table(seed))


def bench_open_addr_hash_family(seed: int = _SEED + 29104) -> dict[str, float]:
    return _finite_blob(bench_open_addr_hash(seed))


def bench_perfect_hash_family(seed: int = _SEED + 29105) -> dict[str, float]:
    return _finite_blob(bench_perfect_hash(seed))

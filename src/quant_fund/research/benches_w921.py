"""Wave-921 distributed-systems-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atomic_bcast import bench_atomic_bcast
from quant_fund.models.avalanche_consensus import bench_avalanche_consensus
from quant_fund.models.honey_badger import bench_honey_badger
from quant_fund.models.isis_bcast import bench_isis_bcast
from quant_fund.models.snowball_consensus import bench_snowball_consensus
from quant_fund.models.virtual_synchrony import bench_virtual_synchrony

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


def bench_virtual_synchrony_family(seed: int = _SEED + 30900) -> dict[str, float]:
    return _finite_blob(bench_virtual_synchrony(seed))


def bench_isis_bcast_family(seed: int = _SEED + 30901) -> dict[str, float]:
    return _finite_blob(bench_isis_bcast(seed))


def bench_atomic_bcast_family(seed: int = _SEED + 30902) -> dict[str, float]:
    return _finite_blob(bench_atomic_bcast(seed))


def bench_honey_badger_family(seed: int = _SEED + 30903) -> dict[str, float]:
    return _finite_blob(bench_honey_badger(seed))


def bench_avalanche_consensus_family(seed: int = _SEED + 30904) -> dict[str, float]:
    return _finite_blob(bench_avalanche_consensus(seed))


def bench_snowball_consensus_family(seed: int = _SEED + 30905) -> dict[str, float]:
    return _finite_blob(bench_snowball_consensus(seed))

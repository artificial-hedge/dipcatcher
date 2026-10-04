"""Wave-1059 history canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ancient_history import bench_ancient_history
from quant_fund.models.economic_history import bench_economic_history
from quant_fund.models.historiography import bench_historiography
from quant_fund.models.intellectual_history import bench_intellectual_history
from quant_fund.models.medieval_history import bench_medieval_history
from quant_fund.models.modern_history import bench_modern_history

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


def bench_historiography_family(seed: int = _SEED + 44700) -> dict[str, float]:
    return _finite_blob(bench_historiography(seed))


def bench_ancient_history_family(seed: int = _SEED + 44701) -> dict[str, float]:
    return _finite_blob(bench_ancient_history(seed))


def bench_medieval_history_family(seed: int = _SEED + 44702) -> dict[str, float]:
    return _finite_blob(bench_medieval_history(seed))


def bench_modern_history_family(seed: int = _SEED + 44703) -> dict[str, float]:
    return _finite_blob(bench_modern_history(seed))


def bench_economic_history_family(seed: int = _SEED + 44704) -> dict[str, float]:
    return _finite_blob(bench_economic_history(seed))


def bench_intellectual_history_family(seed: int = _SEED + 44705) -> dict[str, float]:
    return _finite_blob(bench_intellectual_history(seed))

"""Wave-1127 history-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.digital_history import bench_digital_history
from quant_fund.models.environmental_history import bench_environmental_history
from quant_fund.models.global_history import bench_global_history
from quant_fund.models.maritime_history import bench_maritime_history
from quant_fund.models.oral_history import bench_oral_history
from quant_fund.models.public_history import bench_public_history

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


def bench_oral_history_family(seed: int = _SEED + 51500) -> dict[str, float]:
    return _finite_blob(bench_oral_history(seed))


def bench_public_history_family(seed: int = _SEED + 51501) -> dict[str, float]:
    return _finite_blob(bench_public_history(seed))


def bench_digital_history_family(seed: int = _SEED + 51502) -> dict[str, float]:
    return _finite_blob(bench_digital_history(seed))


def bench_environmental_history_family(seed: int = _SEED + 51503) -> dict[str, float]:
    return _finite_blob(bench_environmental_history(seed))


def bench_global_history_family(seed: int = _SEED + 51504) -> dict[str, float]:
    return _finite_blob(bench_global_history(seed))


def bench_maritime_history_family(seed: int = _SEED + 51505) -> dict[str, float]:
    return _finite_blob(bench_maritime_history(seed))

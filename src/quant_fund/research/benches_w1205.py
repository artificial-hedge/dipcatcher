"""Wave-1205 extreme-environment canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aerospace_medicine import bench_aerospace_medicine
from quant_fund.models.diving_medicine import bench_diving_medicine
from quant_fund.models.high_altitude_medicine import bench_high_altitude_medicine
from quant_fund.models.hyperbaric_oxygen import bench_hyperbaric_oxygen
from quant_fund.models.space_physiology import bench_space_physiology
from quant_fund.models.wilderness_medicine import bench_wilderness_medicine

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


def bench_aerospace_medicine_family(seed: int = _SEED + 59300) -> dict[str, float]:
    return _finite_blob(bench_aerospace_medicine(seed))


def bench_diving_medicine_family(seed: int = _SEED + 59301) -> dict[str, float]:
    return _finite_blob(bench_diving_medicine(seed))


def bench_wilderness_medicine_family(seed: int = _SEED + 59302) -> dict[str, float]:
    return _finite_blob(bench_wilderness_medicine(seed))


def bench_space_physiology_family(seed: int = _SEED + 59303) -> dict[str, float]:
    return _finite_blob(bench_space_physiology(seed))


def bench_hyperbaric_oxygen_family(seed: int = _SEED + 59304) -> dict[str, float]:
    return _finite_blob(bench_hyperbaric_oxygen(seed))


def bench_high_altitude_medicine_family(seed: int = _SEED + 59305) -> dict[str, float]:
    return _finite_blob(bench_high_altitude_medicine(seed))

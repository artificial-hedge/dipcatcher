"""Wave-1125 economics-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.agricultural_economics import bench_agricultural_economics
from quant_fund.models.development_economics import bench_development_economics
from quant_fund.models.energy_economics import bench_energy_economics
from quant_fund.models.environmental_economics import bench_environmental_economics
from quant_fund.models.health_economics import bench_health_economics
from quant_fund.models.urban_economics import bench_urban_economics

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


def bench_development_economics_family(seed: int = _SEED + 51300) -> dict[str, float]:
    return _finite_blob(bench_development_economics(seed))


def bench_environmental_economics_family(seed: int = _SEED + 51301) -> dict[str, float]:
    return _finite_blob(bench_environmental_economics(seed))


def bench_health_economics_family(seed: int = _SEED + 51302) -> dict[str, float]:
    return _finite_blob(bench_health_economics(seed))


def bench_urban_economics_family(seed: int = _SEED + 51303) -> dict[str, float]:
    return _finite_blob(bench_urban_economics(seed))


def bench_agricultural_economics_family(seed: int = _SEED + 51304) -> dict[str, float]:
    return _finite_blob(bench_agricultural_economics(seed))


def bench_energy_economics_family(seed: int = _SEED + 51305) -> dict[str, float]:
    return _finite_blob(bench_energy_economics(seed))

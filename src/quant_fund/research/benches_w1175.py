"""Wave-1175 trades canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.automotive_technology import bench_automotive_technology
from quant_fund.models.carpentry_trades import bench_carpentry_trades
from quant_fund.models.electrical_trades import bench_electrical_trades
from quant_fund.models.plumbing_hvac import bench_plumbing_hvac
from quant_fund.models.refrigeration_technology import bench_refrigeration_technology
from quant_fund.models.welding_technology import bench_welding_technology

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


def bench_electrical_trades_family(seed: int = _SEED + 56300) -> dict[str, float]:
    return _finite_blob(bench_electrical_trades(seed))


def bench_plumbing_hvac_family(seed: int = _SEED + 56301) -> dict[str, float]:
    return _finite_blob(bench_plumbing_hvac(seed))


def bench_welding_technology_family(seed: int = _SEED + 56302) -> dict[str, float]:
    return _finite_blob(bench_welding_technology(seed))


def bench_carpentry_trades_family(seed: int = _SEED + 56303) -> dict[str, float]:
    return _finite_blob(bench_carpentry_trades(seed))


def bench_automotive_technology_family(seed: int = _SEED + 56304) -> dict[str, float]:
    return _finite_blob(bench_automotive_technology(seed))


def bench_refrigeration_technology_family(seed: int = _SEED + 56305) -> dict[str, float]:
    return _finite_blob(bench_refrigeration_technology(seed))

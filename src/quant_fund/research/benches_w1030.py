"""Wave-1030 electrical-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.circuit_analysis import bench_circuit_analysis
from quant_fund.models.control_systems import bench_control_systems
from quant_fund.models.electromagnetics import bench_electromagnetics
from quant_fund.models.power_systems import bench_power_systems
from quant_fund.models.semiconductor import bench_semiconductor
from quant_fund.models.signal_processing2 import bench_signal_processing2

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


def bench_circuit_analysis_family(seed: int = _SEED + 41800) -> dict[str, float]:
    return _finite_blob(bench_circuit_analysis(seed))


def bench_power_systems_family(seed: int = _SEED + 41801) -> dict[str, float]:
    return _finite_blob(bench_power_systems(seed))


def bench_control_systems_family(seed: int = _SEED + 41802) -> dict[str, float]:
    return _finite_blob(bench_control_systems(seed))


def bench_signal_processing2_family(seed: int = _SEED + 41803) -> dict[str, float]:
    return _finite_blob(bench_signal_processing2(seed))


def bench_electromagnetics_family(seed: int = _SEED + 41804) -> dict[str, float]:
    return _finite_blob(bench_electromagnetics(seed))


def bench_semiconductor_family(seed: int = _SEED + 41805) -> dict[str, float]:
    return _finite_blob(bench_semiconductor(seed))

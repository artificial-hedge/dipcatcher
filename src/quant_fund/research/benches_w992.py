"""Wave-992 scattering-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.limiting_absorption import bench_limiting_absorption
from quant_fund.models.radiation_cond import bench_radiation_cond
from quant_fund.models.resonances_thy import bench_resonances_thy
from quant_fund.models.scattering_matrix import bench_scattering_matrix
from quant_fund.models.trace_class_scatt import bench_trace_class_scatt
from quant_fund.models.wave_operators import bench_wave_operators

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


def bench_wave_operators_family(seed: int = _SEED + 38000) -> dict[str, float]:
    return _finite_blob(bench_wave_operators(seed))


def bench_scattering_matrix_family(seed: int = _SEED + 38001) -> dict[str, float]:
    return _finite_blob(bench_scattering_matrix(seed))


def bench_limiting_absorption_family(seed: int = _SEED + 38002) -> dict[str, float]:
    return _finite_blob(bench_limiting_absorption(seed))


def bench_trace_class_scatt_family(seed: int = _SEED + 38003) -> dict[str, float]:
    return _finite_blob(bench_trace_class_scatt(seed))


def bench_resonances_thy_family(seed: int = _SEED + 38004) -> dict[str, float]:
    return _finite_blob(bench_resonances_thy(seed))


def bench_radiation_cond_family(seed: int = _SEED + 38005) -> dict[str, float]:
    return _finite_blob(bench_radiation_cond(seed))

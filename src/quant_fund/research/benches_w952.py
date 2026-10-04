"""Wave-952 spectral-decomposition canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.eigval_bounds import bench_eigval_bounds
from quant_fund.models.power_deflation import bench_power_deflation
from quant_fund.models.qr_iteration import bench_qr_iteration
from quant_fund.models.schur_decomp import bench_schur_decomp
from quant_fund.models.spectral_gap import bench_spectral_gap
from quant_fund.models.spectral_radius import bench_spectral_radius

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


def bench_qr_iteration_family(seed: int = _SEED + 34000) -> dict[str, float]:
    return _finite_blob(bench_qr_iteration(seed))


def bench_power_deflation_family(seed: int = _SEED + 34001) -> dict[str, float]:
    return _finite_blob(bench_power_deflation(seed))


def bench_schur_decomp_family(seed: int = _SEED + 34002) -> dict[str, float]:
    return _finite_blob(bench_schur_decomp(seed))


def bench_eigval_bounds_family(seed: int = _SEED + 34003) -> dict[str, float]:
    return _finite_blob(bench_eigval_bounds(seed))


def bench_spectral_radius_family(seed: int = _SEED + 34004) -> dict[str, float]:
    return _finite_blob(bench_spectral_radius(seed))


def bench_spectral_gap_family(seed: int = _SEED + 34005) -> dict[str, float]:
    return _finite_blob(bench_spectral_gap(seed))

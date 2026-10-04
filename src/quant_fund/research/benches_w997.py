"""Wave-997 dispersive-PDE canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bilinear_estimates import bench_bilinear_estimates
from quant_fund.models.i_method import bench_i_method
from quant_fund.models.kdv_dispersion import bench_kdv_dispersion
from quant_fund.models.local_smoothing import bench_local_smoothing
from quant_fund.models.nls_dispersion import bench_nls_dispersion
from quant_fund.models.strichartz_estimates import bench_strichartz_estimates

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


def bench_nls_dispersion_family(seed: int = _SEED + 38500) -> dict[str, float]:
    return _finite_blob(bench_nls_dispersion(seed))


def bench_kdv_dispersion_family(seed: int = _SEED + 38501) -> dict[str, float]:
    return _finite_blob(bench_kdv_dispersion(seed))


def bench_strichartz_estimates_family(seed: int = _SEED + 38502) -> dict[str, float]:
    return _finite_blob(bench_strichartz_estimates(seed))


def bench_local_smoothing_family(seed: int = _SEED + 38503) -> dict[str, float]:
    return _finite_blob(bench_local_smoothing(seed))


def bench_bilinear_estimates_family(seed: int = _SEED + 38504) -> dict[str, float]:
    return _finite_blob(bench_bilinear_estimates(seed))


def bench_i_method_family(seed: int = _SEED + 38505) -> dict[str, float]:
    return _finite_blob(bench_i_method(seed))

"""Wave-913 interpolation-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.akima_interp import bench_akima_interp
from quant_fund.models.makima_interp import bench_makima_interp
from quant_fund.models.monotone_interp import bench_monotone_interp
from quant_fund.models.pchip_interp import bench_pchip_interp
from quant_fund.models.scattered_interp import bench_scattered_interp
from quant_fund.models.spline_interp import bench_spline_interp

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


def bench_scattered_interp_family(seed: int = _SEED + 30100) -> dict[str, float]:
    return _finite_blob(bench_scattered_interp(seed))


def bench_spline_interp_family(seed: int = _SEED + 30101) -> dict[str, float]:
    return _finite_blob(bench_spline_interp(seed))


def bench_monotone_interp_family(seed: int = _SEED + 30102) -> dict[str, float]:
    return _finite_blob(bench_monotone_interp(seed))


def bench_akima_interp_family(seed: int = _SEED + 30103) -> dict[str, float]:
    return _finite_blob(bench_akima_interp(seed))


def bench_pchip_interp_family(seed: int = _SEED + 30104) -> dict[str, float]:
    return _finite_blob(bench_pchip_interp(seed))


def bench_makima_interp_family(seed: int = _SEED + 30105) -> dict[str, float]:
    return _finite_blob(bench_makima_interp(seed))

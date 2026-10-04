"""Wave-1002 turbulence canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.energy_spectrum import bench_energy_spectrum
from quant_fund.models.intermittency_models import bench_intermittency_models
from quant_fund.models.kolmogorov_theory import bench_kolmogorov_theory
from quant_fund.models.reynolds_decomp import bench_reynolds_decomp
from quant_fund.models.taylor_series_hyp import bench_taylor_series_hyp
from quant_fund.models.wall_turbulence import bench_wall_turbulence

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


def bench_kolmogorov_theory_family(seed: int = _SEED + 39000) -> dict[str, float]:
    return _finite_blob(bench_kolmogorov_theory(seed))


def bench_reynolds_decomp_family(seed: int = _SEED + 39001) -> dict[str, float]:
    return _finite_blob(bench_reynolds_decomp(seed))


def bench_energy_spectrum_family(seed: int = _SEED + 39002) -> dict[str, float]:
    return _finite_blob(bench_energy_spectrum(seed))


def bench_intermittency_models_family(seed: int = _SEED + 39003) -> dict[str, float]:
    return _finite_blob(bench_intermittency_models(seed))


def bench_wall_turbulence_family(seed: int = _SEED + 39004) -> dict[str, float]:
    return _finite_blob(bench_wall_turbulence(seed))


def bench_taylor_series_hyp_family(seed: int = _SEED + 39005) -> dict[str, float]:
    return _finite_blob(bench_taylor_series_hyp(seed))

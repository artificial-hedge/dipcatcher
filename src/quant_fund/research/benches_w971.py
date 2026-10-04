"""Wave-971 noncommutative-geometry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.connes_metric import bench_connes_metric
from quant_fund.models.differential_form_nc import bench_differential_form_nc
from quant_fund.models.geodesic_nc import bench_geodesic_nc
from quant_fund.models.hochschild_cycle import bench_hochschild_cycle
from quant_fund.models.index_pairing import bench_index_pairing
from quant_fund.models.spectral_triple import bench_spectral_triple

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


def bench_spectral_triple_family(seed: int = _SEED + 35900) -> dict[str, float]:
    return _finite_blob(bench_spectral_triple(seed))


def bench_connes_metric_family(seed: int = _SEED + 35901) -> dict[str, float]:
    return _finite_blob(bench_connes_metric(seed))


def bench_index_pairing_family(seed: int = _SEED + 35902) -> dict[str, float]:
    return _finite_blob(bench_index_pairing(seed))


def bench_hochschild_cycle_family(seed: int = _SEED + 35903) -> dict[str, float]:
    return _finite_blob(bench_hochschild_cycle(seed))


def bench_differential_form_nc_family(seed: int = _SEED + 35904) -> dict[str, float]:
    return _finite_blob(bench_differential_form_nc(seed))


def bench_geodesic_nc_family(seed: int = _SEED + 35905) -> dict[str, float]:
    return _finite_blob(bench_geodesic_nc(seed))

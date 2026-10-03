"""Wave-936 nonsmooth-analysis canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bundle_level import bench_bundle_level
from quant_fund.models.clarke_subdiff import bench_clarke_subdiff
from quant_fund.models.epigraph_proj import bench_epigraph_proj
from quant_fund.models.gauge_duality import bench_gauge_duality
from quant_fund.models.gauge_fn import bench_gauge_fn
from quant_fund.models.subdiff_compute import bench_subdiff_compute

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


def bench_subdiff_compute_family(seed: int = _SEED + 32400) -> dict[str, float]:
    return _finite_blob(bench_subdiff_compute(seed))


def bench_epigraph_proj_family(seed: int = _SEED + 32401) -> dict[str, float]:
    return _finite_blob(bench_epigraph_proj(seed))


def bench_gauge_fn_family(seed: int = _SEED + 32402) -> dict[str, float]:
    return _finite_blob(bench_gauge_fn(seed))


def bench_gauge_duality_family(seed: int = _SEED + 32403) -> dict[str, float]:
    return _finite_blob(bench_gauge_duality(seed))


def bench_bundle_level_family(seed: int = _SEED + 32404) -> dict[str, float]:
    return _finite_blob(bench_bundle_level(seed))


def bench_clarke_subdiff_family(seed: int = _SEED + 32405) -> dict[str, float]:
    return _finite_blob(bench_clarke_subdiff(seed))

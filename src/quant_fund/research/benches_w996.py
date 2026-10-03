"""Wave-996 Riemann-Hilbert canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dbar_method import bench_dbar_method
from quant_fund.models.deift_zhou import bench_deift_zhou
from quant_fund.models.fokas_unified import bench_fokas_unified
from quant_fund.models.isomonodromy import bench_isomonodromy
from quant_fund.models.orthogonal_poly_rh import bench_orthogonal_poly_rh
from quant_fund.models.small_norm_rh import bench_small_norm_rh

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


def bench_dbar_method_family(seed: int = _SEED + 38400) -> dict[str, float]:
    return _finite_blob(bench_dbar_method(seed))


def bench_orthogonal_poly_rh_family(seed: int = _SEED + 38401) -> dict[str, float]:
    return _finite_blob(bench_orthogonal_poly_rh(seed))


def bench_isomonodromy_family(seed: int = _SEED + 38402) -> dict[str, float]:
    return _finite_blob(bench_isomonodromy(seed))


def bench_fokas_unified_family(seed: int = _SEED + 38403) -> dict[str, float]:
    return _finite_blob(bench_fokas_unified(seed))


def bench_deift_zhou_family(seed: int = _SEED + 38404) -> dict[str, float]:
    return _finite_blob(bench_deift_zhou(seed))


def bench_small_norm_rh_family(seed: int = _SEED + 38405) -> dict[str, float]:
    return _finite_blob(bench_small_norm_rh(seed))

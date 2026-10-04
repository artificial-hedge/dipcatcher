"""Wave-980 approximation-theory-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.de_boor_stable import bench_de_boor_stable
from quant_fund.models.faber_schauder import bench_faber_schauder
from quant_fund.models.haar_system import bench_haar_system
from quant_fund.models.korovkin_thm import bench_korovkin_thm
from quant_fund.models.walsh_series import bench_walsh_series
from quant_fund.models.whitney_ext import bench_whitney_ext

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


def bench_walsh_series_family(seed: int = _SEED + 36800) -> dict[str, float]:
    return _finite_blob(bench_walsh_series(seed))


def bench_haar_system_family(seed: int = _SEED + 36801) -> dict[str, float]:
    return _finite_blob(bench_haar_system(seed))


def bench_faber_schauder_family(seed: int = _SEED + 36802) -> dict[str, float]:
    return _finite_blob(bench_faber_schauder(seed))


def bench_de_boor_stable_family(seed: int = _SEED + 36803) -> dict[str, float]:
    return _finite_blob(bench_de_boor_stable(seed))


def bench_whitney_ext_family(seed: int = _SEED + 36804) -> dict[str, float]:
    return _finite_blob(bench_whitney_ext(seed))


def bench_korovkin_thm_family(seed: int = _SEED + 36805) -> dict[str, float]:
    return _finite_blob(bench_korovkin_thm(seed))

"""Wave-941 nonsmooth-Newton canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.augmented_lagr import bench_augmented_lagr
from quant_fund.models.ekeland_var import bench_ekeland_var
from quant_fund.models.limiting_subdiff import bench_limiting_subdiff
from quant_fund.models.monteiro_semismooth import bench_monteiro_semismooth
from quant_fund.models.proximal_subdiff import bench_proximal_subdiff
from quant_fund.models.semismooth_newton import bench_semismooth_newton

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


def bench_limiting_subdiff_family(seed: int = _SEED + 32900) -> dict[str, float]:
    return _finite_blob(bench_limiting_subdiff(seed))


def bench_proximal_subdiff_family(seed: int = _SEED + 32901) -> dict[str, float]:
    return _finite_blob(bench_proximal_subdiff(seed))


def bench_ekeland_var_family(seed: int = _SEED + 32902) -> dict[str, float]:
    return _finite_blob(bench_ekeland_var(seed))


def bench_monteiro_semismooth_family(seed: int = _SEED + 32903) -> dict[str, float]:
    return _finite_blob(bench_monteiro_semismooth(seed))


def bench_semismooth_newton_family(seed: int = _SEED + 32904) -> dict[str, float]:
    return _finite_blob(bench_semismooth_newton(seed))


def bench_augmented_lagr_family(seed: int = _SEED + 32905) -> dict[str, float]:
    return _finite_blob(bench_augmented_lagr(seed))

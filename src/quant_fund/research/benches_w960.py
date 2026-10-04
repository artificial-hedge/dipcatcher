"""Wave-960 banach-algebra canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.banach_algebra import bench_banach_algebra
from quant_fund.models.c_star_algebra import bench_c_star_algebra
from quant_fund.models.gelfand_transform import bench_gelfand_transform
from quant_fund.models.holomorphic_calculus import bench_holomorphic_calculus
from quant_fund.models.positive_functional import bench_positive_functional
from quant_fund.models.spectrum_algebra import bench_spectrum_algebra

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


def bench_banach_algebra_family(seed: int = _SEED + 34800) -> dict[str, float]:
    return _finite_blob(bench_banach_algebra(seed))


def bench_gelfand_transform_family(seed: int = _SEED + 34801) -> dict[str, float]:
    return _finite_blob(bench_gelfand_transform(seed))


def bench_c_star_algebra_family(seed: int = _SEED + 34802) -> dict[str, float]:
    return _finite_blob(bench_c_star_algebra(seed))


def bench_spectrum_algebra_family(seed: int = _SEED + 34803) -> dict[str, float]:
    return _finite_blob(bench_spectrum_algebra(seed))


def bench_holomorphic_calculus_family(seed: int = _SEED + 34804) -> dict[str, float]:
    return _finite_blob(bench_holomorphic_calculus(seed))


def bench_positive_functional_family(seed: int = _SEED + 34805) -> dict[str, float]:
    return _finite_blob(bench_positive_functional(seed))

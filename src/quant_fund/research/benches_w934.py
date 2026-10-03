"""Wave-934 convex-analysis-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.inf_convolution import bench_inf_convolution
from quant_fund.models.legendre_transform import bench_legendre_transform
from quant_fund.models.normal_cone import bench_normal_cone
from quant_fund.models.perspective_fn import bench_perspective_fn
from quant_fund.models.polar_cone import bench_polar_cone
from quant_fund.models.support_fn import bench_support_fn

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


def bench_inf_convolution_family(seed: int = _SEED + 32200) -> dict[str, float]:
    return _finite_blob(bench_inf_convolution(seed))


def bench_legendre_transform_family(seed: int = _SEED + 32201) -> dict[str, float]:
    return _finite_blob(bench_legendre_transform(seed))


def bench_support_fn_family(seed: int = _SEED + 32202) -> dict[str, float]:
    return _finite_blob(bench_support_fn(seed))


def bench_perspective_fn_family(seed: int = _SEED + 32203) -> dict[str, float]:
    return _finite_blob(bench_perspective_fn(seed))


def bench_polar_cone_family(seed: int = _SEED + 32204) -> dict[str, float]:
    return _finite_blob(bench_polar_cone(seed))


def bench_normal_cone_family(seed: int = _SEED + 32205) -> dict[str, float]:
    return _finite_blob(bench_normal_cone(seed))

"""Wave-266 graphics-2 benches: software rasterization, shading, texturing, occlusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bump_map import bench_bump_map
from quant_fund.models.mipmap_sample import bench_mipmap_sample
from quant_fund.models.phong_shade import bench_phong_shade
from quant_fund.models.shadow_map import bench_shadow_map
from quant_fund.models.ssao_lite import bench_ssao_lite
from quant_fund.models.triangle_raster import bench_triangle_raster

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_triangle_raster_family(seed: int = _SEED + 1430) -> dict[str, float]:
    return _floats(_finite_blob("triangle_raster", bench_triangle_raster(seed)))


def bench_phong_shade_family(seed: int = _SEED + 1431) -> dict[str, float]:
    return _floats(_finite_blob("phong_shade", bench_phong_shade(seed)))


def bench_mipmap_sample_family(seed: int = _SEED + 1432) -> dict[str, float]:
    return _floats(_finite_blob("mipmap_sample", bench_mipmap_sample(seed)))


def bench_shadow_map_family(seed: int = _SEED + 1433) -> dict[str, float]:
    return _floats(_finite_blob("shadow_map", bench_shadow_map(seed)))


def bench_bump_map_family(seed: int = _SEED + 1434) -> dict[str, float]:
    return _floats(_finite_blob("bump_map", bench_bump_map(seed)))


def bench_ssao_lite_family(seed: int = _SEED + 1435) -> dict[str, float]:
    return _floats(_finite_blob("ssao_lite", bench_ssao_lite(seed)))

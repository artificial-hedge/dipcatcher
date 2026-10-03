"""Wave-293 graphics-3 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.deferred_shade import bench_deferred_shade
from quant_fund.models.env_map import bench_env_map
from quant_fund.models.frustum_cull import bench_frustum_cull
from quant_fund.models.lod_select import bench_lod_select
from quant_fund.models.sdf_raymarch import bench_sdf_raymarch
from quant_fund.models.shadow_pcf import bench_shadow_pcf

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


def bench_deferred_shade_family(seed: int = _SEED + 1664) -> dict[str, float]:
    return _floats(_finite_blob("deferred_shade", bench_deferred_shade(seed)))


def bench_sdf_raymarch_family(seed: int = _SEED + 1665) -> dict[str, float]:
    return _floats(_finite_blob("sdf_raymarch", bench_sdf_raymarch(seed)))


def bench_frustum_cull_family(seed: int = _SEED + 1666) -> dict[str, float]:
    return _floats(_finite_blob("frustum_cull", bench_frustum_cull(seed)))


def bench_lod_select_family(seed: int = _SEED + 1667) -> dict[str, float]:
    return _floats(_finite_blob("lod_select", bench_lod_select(seed)))


def bench_env_map_family(seed: int = _SEED + 1668) -> dict[str, float]:
    return _floats(_finite_blob("env_map", bench_env_map(seed)))


def bench_shadow_pcf_family(seed: int = _SEED + 1669) -> dict[str, float]:
    return _floats(_finite_blob("shadow_pcf", bench_shadow_pcf(seed)))

"""Wave-493 tropical-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.berkovich_an import bench_berkovich_an
from quant_fund.models.mikhalkin import bench_mikhalkin
from quant_fund.models.skeleton_trop import bench_skeleton_trop
from quant_fund.models.tropical_curve import bench_tropical_curve
from quant_fund.models.tropical_cycle import bench_tropical_cycle
from quant_fund.models.tropical_poly import bench_tropical_poly

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_tropical_poly_family(seed: int = _SEED + 2864) -> dict[str, float]:
    return _floats(_finite_blob("tropical_poly", bench_tropical_poly(seed)))


def bench_berkovich_an_family(seed: int = _SEED + 2865) -> dict[str, float]:
    return _floats(_finite_blob("berkovich_an", bench_berkovich_an(seed)))


def bench_skeleton_trop_family(seed: int = _SEED + 2866) -> dict[str, float]:
    return _floats(_finite_blob("skeleton_trop", bench_skeleton_trop(seed)))


def bench_tropical_curve_family(seed: int = _SEED + 2867) -> dict[str, float]:
    return _floats(_finite_blob("tropical_curve", bench_tropical_curve(seed)))


def bench_mikhalkin_family(seed: int = _SEED + 2868) -> dict[str, float]:
    return _floats(_finite_blob("mikhalkin", bench_mikhalkin(seed)))


def bench_tropical_cycle_family(seed: int = _SEED + 2869) -> dict[str, float]:
    return _floats(_finite_blob("tropical_cycle", bench_tropical_cycle(seed)))

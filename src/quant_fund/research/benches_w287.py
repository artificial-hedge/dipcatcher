"""Wave-287 differential-geometry canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.christoffel import bench_christoffel
from quant_fund.models.first_ff import bench_first_ff
from quant_fund.models.frenet_frame import bench_frenet_frame
from quant_fund.models.gauss_curve import bench_gauss_curve
from quant_fund.models.geodesic_sphere import bench_geodesic_sphere
from quant_fund.models.surf_area import bench_surf_area

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


def bench_first_ff_family(seed: int = _SEED + 1628) -> dict[str, float]:
    return _floats(_finite_blob("first_ff", bench_first_ff(seed)))


def bench_gauss_curve_family(seed: int = _SEED + 1629) -> dict[str, float]:
    return _floats(_finite_blob("gauss_curve", bench_gauss_curve(seed)))


def bench_frenet_frame_family(seed: int = _SEED + 1630) -> dict[str, float]:
    return _floats(_finite_blob("frenet_frame", bench_frenet_frame(seed)))


def bench_christoffel_family(seed: int = _SEED + 1631) -> dict[str, float]:
    return _floats(_finite_blob("christoffel", bench_christoffel(seed)))


def bench_geodesic_sphere_family(seed: int = _SEED + 1632) -> dict[str, float]:
    return _floats(_finite_blob("geodesic_sphere", bench_geodesic_sphere(seed)))


def bench_surf_area_family(seed: int = _SEED + 1633) -> dict[str, float]:
    return _floats(_finite_blob("surf_area", bench_surf_area(seed)))

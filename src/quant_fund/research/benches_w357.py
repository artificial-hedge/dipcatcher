"""Wave-357 differential-geometry-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.connection_form import bench_connection_form
from quant_fund.models.gauss_bonnet import bench_gauss_bonnet
from quant_fund.models.geodesic_eq import bench_geodesic_eq
from quant_fund.models.holonomy import bench_holonomy
from quant_fund.models.parallel_transport import bench_parallel_transport
from quant_fund.models.sectional_curv import bench_sectional_curv

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


def bench_connection_form_family(seed: int = _SEED + 2049) -> dict[str, float]:
    return _floats(_finite_blob("connection_form", bench_connection_form(seed)))


def bench_parallel_transport_family(seed: int = _SEED + 2050) -> dict[str, float]:
    return _floats(_finite_blob("parallel_transport", bench_parallel_transport(seed)))


def bench_holonomy_family(seed: int = _SEED + 2051) -> dict[str, float]:
    return _floats(_finite_blob("holonomy", bench_holonomy(seed)))


def bench_gauss_bonnet_family(seed: int = _SEED + 2052) -> dict[str, float]:
    return _floats(_finite_blob("gauss_bonnet", bench_gauss_bonnet(seed)))


def bench_geodesic_eq_family(seed: int = _SEED + 2053) -> dict[str, float]:
    return _floats(_finite_blob("geodesic_eq", bench_geodesic_eq(seed)))


def bench_sectional_curv_family(seed: int = _SEED + 2054) -> dict[str, float]:
    return _floats(_finite_blob("sectional_curv", bench_sectional_curv(seed)))

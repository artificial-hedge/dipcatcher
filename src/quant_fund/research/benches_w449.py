"""Wave-449 analytic-geometry-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adic_generic import bench_adic_generic
from quant_fund.models.dagger_space import bench_dagger_space
from quant_fund.models.fargues_curve import bench_fargues_curve
from quant_fund.models.huber_ring import bench_huber_ring
from quant_fund.models.prism_site import bench_prism_site
from quant_fund.models.witt_perfect import bench_witt_perfect

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


def bench_dagger_space_family(seed: int = _SEED + 2600) -> dict[str, float]:
    return _floats(_finite_blob("dagger_space", bench_dagger_space(seed)))


def bench_huber_ring_family(seed: int = _SEED + 2601) -> dict[str, float]:
    return _floats(_finite_blob("huber_ring", bench_huber_ring(seed)))


def bench_adic_generic_family(seed: int = _SEED + 2602) -> dict[str, float]:
    return _floats(_finite_blob("adic_generic", bench_adic_generic(seed)))


def bench_witt_perfect_family(seed: int = _SEED + 2603) -> dict[str, float]:
    return _floats(_finite_blob("witt_perfect", bench_witt_perfect(seed)))


def bench_fargues_curve_family(seed: int = _SEED + 2604) -> dict[str, float]:
    return _floats(_finite_blob("fargues_curve", bench_fargues_curve(seed)))


def bench_prism_site_family(seed: int = _SEED + 2605) -> dict[str, float]:
    return _floats(_finite_blob("prism_site", bench_prism_site(seed)))

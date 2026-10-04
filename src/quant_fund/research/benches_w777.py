"""Wave-777 heavy-traffic bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.diffusion_approx import (
    bench_diffusion_approx,
)
from quant_fund.models.fluid_limit import bench_fluid_limit
from quant_fund.models.halfin_whitt import bench_halfin_whitt
from quant_fund.models.heavy_traffic import bench_heavy_traffic
from quant_fund.models.kingman_bound import (
    bench_kingman_bound,
)
from quant_fund.models.qed_regime import bench_qed_regime

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


def bench_fluid_limit_family(
    seed: int = _SEED + 16600,
) -> dict[str, float]:
    return _floats(_finite_blob("fluid_limit", bench_fluid_limit(seed)))


def bench_heavy_traffic_family(
    seed: int = _SEED + 16601,
) -> dict[str, float]:
    return _floats(_finite_blob("heavy_traffic", bench_heavy_traffic(seed)))


def bench_diffusion_approx_family(
    seed: int = _SEED + 16602,
) -> dict[str, float]:
    return _floats(_finite_blob("diffusion_approx", bench_diffusion_approx(seed)))


def bench_kingman_bound_family(
    seed: int = _SEED + 16603,
) -> dict[str, float]:
    return _floats(_finite_blob("kingman_bound", bench_kingman_bound(seed)))


def bench_halfin_whitt_family(
    seed: int = _SEED + 16604,
) -> dict[str, float]:
    return _floats(_finite_blob("halfin_whitt", bench_halfin_whitt(seed)))


def bench_qed_regime_family(
    seed: int = _SEED + 16605,
) -> dict[str, float]:
    return _floats(_finite_blob("qed_regime", bench_qed_regime(seed)))

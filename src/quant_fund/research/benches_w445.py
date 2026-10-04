"""Wave-445 six-functor bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.base_change import bench_base_change
from quant_fund.models.constructible import bench_constructible
from quant_fund.models.perverse_sh import bench_perverse_sh
from quant_fund.models.projection_frm import bench_projection_frm
from quant_fund.models.six_functors import bench_six_functors
from quant_fund.models.verdier_dual import bench_verdier_dual

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


def bench_six_functors_family(seed: int = _SEED + 2576) -> dict[str, float]:
    return _floats(_finite_blob("six_functors", bench_six_functors(seed)))


def bench_base_change_family(seed: int = _SEED + 2577) -> dict[str, float]:
    return _floats(_finite_blob("base_change", bench_base_change(seed)))


def bench_projection_frm_family(seed: int = _SEED + 2578) -> dict[str, float]:
    return _floats(_finite_blob("projection_frm", bench_projection_frm(seed)))


def bench_verdier_dual_family(seed: int = _SEED + 2579) -> dict[str, float]:
    return _floats(_finite_blob("verdier_dual", bench_verdier_dual(seed)))


def bench_constructible_family(seed: int = _SEED + 2580) -> dict[str, float]:
    return _floats(_finite_blob("constructible", bench_constructible(seed)))


def bench_perverse_sh_family(seed: int = _SEED + 2581) -> dict[str, float]:
    return _floats(_finite_blob("perverse_sh", bench_perverse_sh(seed)))

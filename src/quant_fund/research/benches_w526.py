"""Wave-526 complex-dynamics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.douady_hubbard import bench_douady_hubbard
from quant_fund.models.fatou_set import bench_fatou_set
from quant_fund.models.julia_set import bench_julia_set
from quant_fund.models.mandelbrot_set import bench_mandelbrot_set
from quant_fund.models.parabolic_impl import bench_parabolic_impl
from quant_fund.models.sullivan_no_wander import bench_sullivan_no_wander

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


def bench_julia_set_family(seed: int = _SEED + 3062) -> dict[str, float]:
    return _floats(_finite_blob("julia_set", bench_julia_set(seed)))


def bench_mandelbrot_set_family(seed: int = _SEED + 3063) -> dict[str, float]:
    return _floats(_finite_blob("mandelbrot_set", bench_mandelbrot_set(seed)))


def bench_fatou_set_family(seed: int = _SEED + 3064) -> dict[str, float]:
    return _floats(_finite_blob("fatou_set", bench_fatou_set(seed)))


def bench_sullivan_no_wander_family(
    seed: int = _SEED + 3065,
) -> dict[str, float]:
    return _floats(_finite_blob("sullivan_no_wander", bench_sullivan_no_wander(seed)))


def bench_douady_hubbard_family(seed: int = _SEED + 3066) -> dict[str, float]:
    return _floats(_finite_blob("douady_hubbard", bench_douady_hubbard(seed)))


def bench_parabolic_impl_family(seed: int = _SEED + 3067) -> dict[str, float]:
    return _floats(_finite_blob("parabolic_impl", bench_parabolic_impl(seed)))

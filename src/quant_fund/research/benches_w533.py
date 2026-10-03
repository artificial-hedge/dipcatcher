"""Wave-533 geometric-measure-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.besicovitch import bench_besicovitch
from quant_fund.models.density_thm import bench_density_thm
from quant_fund.models.marstrand import bench_marstrand
from quant_fund.models.preiss_rect import bench_preiss_rect
from quant_fund.models.rectifiability import bench_rectifiability
from quant_fund.models.tangent_measure import bench_tangent_measure

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


def bench_rectifiability_family(seed: int = _SEED + 3104) -> dict[str, float]:
    return _floats(_finite_blob("rectifiability", bench_rectifiability(seed)))


def bench_tangent_measure_family(seed: int = _SEED + 3105) -> dict[str, float]:
    return _floats(_finite_blob("tangent_measure", bench_tangent_measure(seed)))


def bench_density_thm_family(seed: int = _SEED + 3106) -> dict[str, float]:
    return _floats(_finite_blob("density_thm", bench_density_thm(seed)))


def bench_marstrand_family(seed: int = _SEED + 3107) -> dict[str, float]:
    return _floats(_finite_blob("marstrand", bench_marstrand(seed)))


def bench_besicovitch_family(seed: int = _SEED + 3108) -> dict[str, float]:
    return _floats(_finite_blob("besicovitch", bench_besicovitch(seed)))


def bench_preiss_rect_family(seed: int = _SEED + 3109) -> dict[str, float]:
    return _floats(_finite_blob("preiss_rect", bench_preiss_rect(seed)))

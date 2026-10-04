"""Wave-529 KAM/Aubry-Mather bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arnold_diff import bench_arnold_diff
from quant_fund.models.aubry_mather import bench_aubry_mather
from quant_fund.models.cantorus import bench_cantorus
from quant_fund.models.greene_crit import bench_greene_crit
from quant_fund.models.kam_theorem import bench_kam_theorem
from quant_fund.models.twist_map import bench_twist_map

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


def bench_kam_theorem_family(seed: int = _SEED + 3080) -> dict[str, float]:
    return _floats(_finite_blob("kam_theorem", bench_kam_theorem(seed)))


def bench_aubry_mather_family(seed: int = _SEED + 3081) -> dict[str, float]:
    return _floats(_finite_blob("aubry_mather", bench_aubry_mather(seed)))


def bench_twist_map_family(seed: int = _SEED + 3082) -> dict[str, float]:
    return _floats(_finite_blob("twist_map", bench_twist_map(seed)))


def bench_cantorus_family(seed: int = _SEED + 3083) -> dict[str, float]:
    return _floats(_finite_blob("cantorus", bench_cantorus(seed)))


def bench_greene_crit_family(seed: int = _SEED + 3084) -> dict[str, float]:
    return _floats(_finite_blob("greene_crit", bench_greene_crit(seed)))


def bench_arnold_diff_family(seed: int = _SEED + 3085) -> dict[str, float]:
    return _floats(_finite_blob("arnold_diff", bench_arnold_diff(seed)))

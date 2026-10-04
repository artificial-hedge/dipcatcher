"""Wave-471 p-adic-geometry-2/perfectoid bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.banach_colmez import bench_banach_colmez
from quant_fund.models.breuil_kisin import bench_breuil_kisin
from quant_fund.models.diamond_geo import bench_diamond_geo
from quant_fund.models.drinfeld_tower import bench_drinfeld_tower
from quant_fund.models.integral_padic import bench_integral_padic
from quant_fund.models.perfectoid2 import bench_perfectoid2

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


def bench_perfectoid2_family(seed: int = _SEED + 2732) -> dict[str, float]:
    return _floats(_finite_blob("perfectoid2", bench_perfectoid2(seed)))


def bench_diamond_geo_family(seed: int = _SEED + 2733) -> dict[str, float]:
    return _floats(_finite_blob("diamond_geo", bench_diamond_geo(seed)))


def bench_integral_padic_family(seed: int = _SEED + 2734) -> dict[str, float]:
    return _floats(_finite_blob("integral_padic", bench_integral_padic(seed)))


def bench_breuil_kisin_family(seed: int = _SEED + 2735) -> dict[str, float]:
    return _floats(_finite_blob("breuil_kisin", bench_breuil_kisin(seed)))


def bench_banach_colmez_family(seed: int = _SEED + 2736) -> dict[str, float]:
    return _floats(_finite_blob("banach_colmez", bench_banach_colmez(seed)))


def bench_drinfeld_tower_family(seed: int = _SEED + 2737) -> dict[str, float]:
    return _floats(_finite_blob("drinfeld_tower", bench_drinfeld_tower(seed)))

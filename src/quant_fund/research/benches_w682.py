"""Wave-682 motivic-18 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_crystal import bench_motivic_crystal
from quant_fund.models.motivic_cycle import bench_motivic_cycle
from quant_fund.models.motivic_etale import bench_motivic_etale
from quant_fund.models.motivic_prism import bench_motivic_prism
from quant_fund.models.motivic_sphere3 import bench_motivic_sphere3
from quant_fund.models.motivic_tower import bench_motivic_tower

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


def bench_motivic_tower_family(
    seed: int = _SEED + 7100,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_tower", bench_motivic_tower(seed)))


def bench_motivic_sphere3_family(
    seed: int = _SEED + 7101,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_sphere3", bench_motivic_sphere3(seed)))


def bench_motivic_etale_family(
    seed: int = _SEED + 7102,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_etale", bench_motivic_etale(seed)))


def bench_motivic_crystal_family(
    seed: int = _SEED + 7103,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_crystal", bench_motivic_crystal(seed)))


def bench_motivic_prism_family(
    seed: int = _SEED + 7104,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_prism", bench_motivic_prism(seed)))


def bench_motivic_cycle_family(
    seed: int = _SEED + 7105,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_cycle", bench_motivic_cycle(seed)))

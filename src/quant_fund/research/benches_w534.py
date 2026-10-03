"""Wave-534 potential-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.balayage import bench_balayage
from quant_fund.models.capacity_theory import bench_capacity_theory
from quant_fund.models.fine_topology import bench_fine_topology
from quant_fund.models.green_fn import bench_green_fn
from quant_fund.models.harmonic_fn import bench_harmonic_fn
from quant_fund.models.potential_thy import bench_potential_thy

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


def bench_harmonic_fn_family(seed: int = _SEED + 3110) -> dict[str, float]:
    return _floats(_finite_blob("harmonic_fn", bench_harmonic_fn(seed)))


def bench_potential_thy_family(seed: int = _SEED + 3111) -> dict[str, float]:
    return _floats(_finite_blob("potential_thy", bench_potential_thy(seed)))


def bench_capacity_theory_family(
    seed: int = _SEED + 3112,
) -> dict[str, float]:
    return _floats(_finite_blob("capacity_theory", bench_capacity_theory(seed)))


def bench_balayage_family(seed: int = _SEED + 3113) -> dict[str, float]:
    return _floats(_finite_blob("balayage", bench_balayage(seed)))


def bench_green_fn_family(seed: int = _SEED + 3114) -> dict[str, float]:
    return _floats(_finite_blob("green_fn", bench_green_fn(seed)))


def bench_fine_topology_family(seed: int = _SEED + 3115) -> dict[str, float]:
    return _floats(_finite_blob("fine_topology", bench_fine_topology(seed)))

"""Wave-420 algebraic-NT-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dedekind_zeta import bench_dedekind_zeta
from quant_fund.models.dirichlet_unit import bench_dirichlet_unit
from quant_fund.models.ideal_class import bench_ideal_class
from quant_fund.models.minkowski_bound import bench_minkowski_bound
from quant_fund.models.regulator import bench_regulator
from quant_fund.models.splitting_prime import bench_splitting_prime

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


def bench_dirichlet_unit_family(
    seed: int = _SEED + 2426,
) -> dict[str, float]:
    return _floats(_finite_blob("dirichlet_unit", bench_dirichlet_unit(seed)))


def bench_regulator_family(
    seed: int = _SEED + 2427,
) -> dict[str, float]:
    return _floats(_finite_blob("regulator", bench_regulator(seed)))


def bench_ideal_class_family(
    seed: int = _SEED + 2428,
) -> dict[str, float]:
    return _floats(_finite_blob("ideal_class", bench_ideal_class(seed)))


def bench_minkowski_bound_family(
    seed: int = _SEED + 2429,
) -> dict[str, float]:
    return _floats(_finite_blob("minkowski_bound", bench_minkowski_bound(seed)))


def bench_dedekind_zeta_family(
    seed: int = _SEED + 2430,
) -> dict[str, float]:
    return _floats(_finite_blob("dedekind_zeta", bench_dedekind_zeta(seed)))


def bench_splitting_prime_family(
    seed: int = _SEED + 2431,
) -> dict[str, float]:
    return _floats(_finite_blob("splitting_prime", bench_splitting_prime(seed)))

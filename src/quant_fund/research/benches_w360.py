"""Wave-360 PDE-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.energy_method import bench_energy_method
from quant_fund.models.fundamental_laplace import bench_fundamental_laplace
from quant_fund.models.heat_kernel import bench_heat_kernel
from quant_fund.models.maximum_principle import bench_maximum_principle
from quant_fund.models.wave_dalembert import bench_wave_dalembert
from quant_fund.models.weak_solution import bench_weak_solution

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


def bench_energy_method_family(seed: int = _SEED + 2067) -> dict[str, float]:
    return _floats(_finite_blob("energy_method", bench_energy_method(seed)))


def bench_maximum_principle_family(seed: int = _SEED + 2068) -> dict[str, float]:
    return _floats(_finite_blob("maximum_principle", bench_maximum_principle(seed)))


def bench_heat_kernel_family(seed: int = _SEED + 2069) -> dict[str, float]:
    return _floats(_finite_blob("heat_kernel", bench_heat_kernel(seed)))


def bench_wave_dalembert_family(seed: int = _SEED + 2070) -> dict[str, float]:
    return _floats(_finite_blob("wave_dalembert", bench_wave_dalembert(seed)))


def bench_weak_solution_family(seed: int = _SEED + 2071) -> dict[str, float]:
    return _floats(_finite_blob("weak_solution", bench_weak_solution(seed)))


def bench_fundamental_laplace_family(seed: int = _SEED + 2072) -> dict[str, float]:
    return _floats(_finite_blob("fundamental_laplace", bench_fundamental_laplace(seed)))

"""Wave-435 derived-schemes bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_fiber import bench_derived_fiber
from quant_fund.models.derived_scheme import bench_derived_scheme
from quant_fund.models.quasi_coherent import bench_quasi_coherent
from quant_fund.models.shifted_symplectic import (
    bench_shifted_symplectic,
)
from quant_fund.models.spectral_scheme import bench_spectral_scheme
from quant_fund.models.virtual_class import bench_virtual_class

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


def bench_derived_scheme_family(
    seed: int = _SEED + 2516,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_scheme", bench_derived_scheme(seed)))


def bench_quasi_coherent_family(
    seed: int = _SEED + 2517,
) -> dict[str, float]:
    return _floats(_finite_blob("quasi_coherent", bench_quasi_coherent(seed)))


def bench_derived_fiber_family(
    seed: int = _SEED + 2518,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_fiber", bench_derived_fiber(seed)))


def bench_spectral_scheme_family(
    seed: int = _SEED + 2519,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_scheme", bench_spectral_scheme(seed)))


def bench_virtual_class_family(
    seed: int = _SEED + 2520,
) -> dict[str, float]:
    return _floats(_finite_blob("virtual_class", bench_virtual_class(seed)))


def bench_shifted_symplectic_family(
    seed: int = _SEED + 2521,
) -> dict[str, float]:
    return _floats(_finite_blob("shifted_symplectic", bench_shifted_symplectic(seed)))

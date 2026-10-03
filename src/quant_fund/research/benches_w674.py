"""Wave-674 spectral-AG-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.analytic_spec import bench_analytic_spec
from quant_fund.models.derived_k3 import bench_derived_k3
from quant_fund.models.equivariant_spec import (
    bench_equivariant_spec,
)
from quant_fund.models.graded_spec import bench_graded_spec
from quant_fund.models.spectral_curve import bench_spectral_curve
from quant_fund.models.spectral_gm import bench_spectral_gm

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


def bench_derived_k3_family(
    seed: int = _SEED + 6300,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_k3", bench_derived_k3(seed)))


def bench_spectral_gm_family(
    seed: int = _SEED + 6301,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_gm", bench_spectral_gm(seed)))


def bench_analytic_spec_family(
    seed: int = _SEED + 6302,
) -> dict[str, float]:
    return _floats(_finite_blob("analytic_spec", bench_analytic_spec(seed)))


def bench_graded_spec_family(
    seed: int = _SEED + 6303,
) -> dict[str, float]:
    return _floats(_finite_blob("graded_spec", bench_graded_spec(seed)))


def bench_equivariant_spec_family(
    seed: int = _SEED + 6304,
) -> dict[str, float]:
    return _floats(_finite_blob("equivariant_spec", bench_equivariant_spec(seed)))


def bench_spectral_curve_family(
    seed: int = _SEED + 6305,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_curve", bench_spectral_curve(seed)))

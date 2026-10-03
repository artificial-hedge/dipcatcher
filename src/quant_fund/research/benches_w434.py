"""Wave-434 Langlands-toy bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.automorphic_rep import (
    bench_automorphic_rep,
)
from quant_fund.models.eisenstein_srs import bench_eisenstein_srs
from quant_fund.models.fourier_coeff import bench_fourier_coeff
from quant_fund.models.hecke_operator import bench_hecke_operator
from quant_fund.models.langlands_dual import bench_langlands_dual
from quant_fund.models.satake_iso import bench_satake_iso

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


def bench_satake_iso_family(
    seed: int = _SEED + 2510,
) -> dict[str, float]:
    return _floats(_finite_blob("satake_iso", bench_satake_iso(seed)))


def bench_hecke_operator_family(
    seed: int = _SEED + 2511,
) -> dict[str, float]:
    return _floats(_finite_blob("hecke_operator", bench_hecke_operator(seed)))


def bench_langlands_dual_family(
    seed: int = _SEED + 2512,
) -> dict[str, float]:
    return _floats(_finite_blob("langlands_dual", bench_langlands_dual(seed)))


def bench_eisenstein_srs_family(
    seed: int = _SEED + 2513,
) -> dict[str, float]:
    return _floats(_finite_blob("eisenstein_srs", bench_eisenstein_srs(seed)))


def bench_automorphic_rep_family(
    seed: int = _SEED + 2514,
) -> dict[str, float]:
    return _floats(_finite_blob("automorphic_rep", bench_automorphic_rep(seed)))


def bench_fourier_coeff_family(
    seed: int = _SEED + 2515,
) -> dict[str, float]:
    return _floats(_finite_blob("fourier_coeff", bench_fourier_coeff(seed)))

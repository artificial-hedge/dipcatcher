"""Wave-359 harmonic-analysis canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fejer_kernel import bench_fejer_kernel
from quant_fund.models.fourier_multiplier import bench_fourier_multiplier
from quant_fund.models.plancherel import bench_plancherel
from quant_fund.models.poisson_summation import bench_poisson_summation
from quant_fund.models.sobolev_embed import bench_sobolev_embed
from quant_fund.models.uncertainty import bench_uncertainty

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


def bench_plancherel_family(seed: int = _SEED + 2061) -> dict[str, float]:
    return _floats(_finite_blob("plancherel", bench_plancherel(seed)))


def bench_poisson_summation_family(seed: int = _SEED + 2062) -> dict[str, float]:
    return _floats(_finite_blob("poisson_summation", bench_poisson_summation(seed)))


def bench_fejer_kernel_family(seed: int = _SEED + 2063) -> dict[str, float]:
    return _floats(_finite_blob("fejer_kernel", bench_fejer_kernel(seed)))


def bench_uncertainty_family(seed: int = _SEED + 2064) -> dict[str, float]:
    return _floats(_finite_blob("uncertainty", bench_uncertainty(seed)))


def bench_fourier_multiplier_family(seed: int = _SEED + 2065) -> dict[str, float]:
    return _floats(_finite_blob("fourier_multiplier", bench_fourier_multiplier(seed)))


def bench_sobolev_embed_family(seed: int = _SEED + 2066) -> dict[str, float]:
    return _floats(_finite_blob("sobolev_embed", bench_sobolev_embed(seed)))

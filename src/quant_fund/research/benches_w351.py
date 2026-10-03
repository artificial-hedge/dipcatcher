"""Wave-351 functional-analysis canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.banach_fixed import bench_banach_fixed
from quant_fund.models.compact_operator import bench_compact_operator
from quant_fund.models.fourier_finite import bench_fourier_finite
from quant_fund.models.gram_schmidt import bench_gram_schmidt
from quant_fund.models.lp_duality import bench_lp_duality
from quant_fund.models.spectral_theorem import bench_spectral_theorem

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


def bench_banach_fixed_family(seed: int = _SEED + 2013) -> dict[str, float]:
    return _floats(_finite_blob("banach_fixed", bench_banach_fixed(seed)))


def bench_spectral_theorem_family(seed: int = _SEED + 2014) -> dict[str, float]:
    return _floats(_finite_blob("spectral_theorem", bench_spectral_theorem(seed)))


def bench_lp_duality_family(seed: int = _SEED + 2015) -> dict[str, float]:
    return _floats(_finite_blob("lp_duality", bench_lp_duality(seed)))


def bench_fourier_finite_family(seed: int = _SEED + 2016) -> dict[str, float]:
    return _floats(_finite_blob("fourier_finite", bench_fourier_finite(seed)))


def bench_compact_operator_family(seed: int = _SEED + 2017) -> dict[str, float]:
    return _floats(_finite_blob("compact_operator", bench_compact_operator(seed)))


def bench_gram_schmidt_family(seed: int = _SEED + 2018) -> dict[str, float]:
    return _floats(_finite_blob("gram_schmidt", bench_gram_schmidt(seed)))

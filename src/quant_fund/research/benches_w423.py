"""Wave-423 homological-algebra-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adams_diff import bench_adams_diff
from quant_fund.models.cartan_eilenberg import bench_cartan_eilenberg
from quant_fund.models.deriv_hom import bench_deriv_hom
from quant_fund.models.groth_spectral import bench_groth_spectral
from quant_fund.models.hypercohom import bench_hypercohom
from quant_fund.models.serre_ss2 import bench_serre_ss2

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


def bench_groth_spectral_family(
    seed: int = _SEED + 2444,
) -> dict[str, float]:
    return _floats(_finite_blob("groth_spectral", bench_groth_spectral(seed)))


def bench_serre_ss2_family(
    seed: int = _SEED + 2445,
) -> dict[str, float]:
    return _floats(_finite_blob("serre_ss2", bench_serre_ss2(seed)))


def bench_hypercohom_family(
    seed: int = _SEED + 2446,
) -> dict[str, float]:
    return _floats(_finite_blob("hypercohom", bench_hypercohom(seed)))


def bench_deriv_hom_family(
    seed: int = _SEED + 2447,
) -> dict[str, float]:
    return _floats(_finite_blob("deriv_hom", bench_deriv_hom(seed)))


def bench_cartan_eilenberg_family(
    seed: int = _SEED + 2448,
) -> dict[str, float]:
    return _floats(_finite_blob("cartan_eilenberg", bench_cartan_eilenberg(seed)))


def bench_adams_diff_family(
    seed: int = _SEED + 2449,
) -> dict[str, float]:
    return _floats(_finite_blob("adams_diff", bench_adams_diff(seed)))

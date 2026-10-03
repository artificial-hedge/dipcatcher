"""Wave-406 homological-algebra-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.functor_derived import bench_functor_derived
from quant_fund.models.hopf_algebra2 import bench_hopf_algebra2
from quant_fund.models.kunneth import bench_kunneth
from quant_fund.models.leray_hirsch import bench_leray_hirsch
from quant_fund.models.poincare_duality2 import bench_poincare_duality2
from quant_fund.models.universal_coeff import bench_universal_coeff

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


def bench_poincare_duality2_family(
    seed: int = _SEED + 2342,
) -> dict[str, float]:
    return _floats(_finite_blob("poincare_duality2", bench_poincare_duality2(seed)))


def bench_universal_coeff_family(
    seed: int = _SEED + 2343,
) -> dict[str, float]:
    return _floats(_finite_blob("universal_coeff", bench_universal_coeff(seed)))


def bench_kunneth_family(seed: int = _SEED + 2344) -> dict[str, float]:
    return _floats(_finite_blob("kunneth", bench_kunneth(seed)))


def bench_leray_hirsch_family(seed: int = _SEED + 2345) -> dict[str, float]:
    return _floats(_finite_blob("leray_hirsch", bench_leray_hirsch(seed)))


def bench_hopf_algebra2_family(
    seed: int = _SEED + 2346,
) -> dict[str, float]:
    return _floats(_finite_blob("hopf_algebra2", bench_hopf_algebra2(seed)))


def bench_functor_derived_family(
    seed: int = _SEED + 2347,
) -> dict[str, float]:
    return _floats(_finite_blob("functor_derived", bench_functor_derived(seed)))

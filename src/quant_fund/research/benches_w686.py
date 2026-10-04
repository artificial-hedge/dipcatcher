"""Wave-686 higher-algebra-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.braces_e4 import bench_braces_e4
from quant_fund.models.centralizer_alg import (
    bench_centralizer_alg,
)
from quant_fund.models.delooping2 import bench_delooping2
from quant_fund.models.e4_algebra import bench_e4_algebra
from quant_fund.models.factorization_hom2 import (
    bench_factorization_hom2,
)
from quant_fund.models.koszul_duality2 import (
    bench_koszul_duality2,
)

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


def bench_e4_algebra_family(
    seed: int = _SEED + 7500,
) -> dict[str, float]:
    return _floats(_finite_blob("e4_algebra", bench_e4_algebra(seed)))


def bench_centralizer_alg_family(
    seed: int = _SEED + 7501,
) -> dict[str, float]:
    return _floats(_finite_blob("centralizer_alg", bench_centralizer_alg(seed)))


def bench_delooping2_family(
    seed: int = _SEED + 7502,
) -> dict[str, float]:
    return _floats(_finite_blob("delooping2", bench_delooping2(seed)))


def bench_factorization_hom2_family(
    seed: int = _SEED + 7503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "factorization_hom2",
            bench_factorization_hom2(seed),
        )
    )


def bench_koszul_duality2_family(
    seed: int = _SEED + 7504,
) -> dict[str, float]:
    return _floats(_finite_blob("koszul_duality2", bench_koszul_duality2(seed)))


def bench_braces_e4_family(
    seed: int = _SEED + 7505,
) -> dict[str, float]:
    return _floats(_finite_blob("braces_e4", bench_braces_e4(seed)))

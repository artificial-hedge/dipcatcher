"""Wave-408 representation-theory-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brauer_alg import bench_brauer_alg
from quant_fund.models.bz_category import bench_bz_category
from quant_fund.models.casimir_op import bench_casimir_op
from quant_fund.models.hecke_alg import bench_hecke_alg
from quant_fund.models.schur_functor import bench_schur_functor
from quant_fund.models.weight_space import bench_weight_space

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


def bench_schur_functor_family(
    seed: int = _SEED + 2354,
) -> dict[str, float]:
    return _floats(_finite_blob("schur_functor", bench_schur_functor(seed)))


def bench_brauer_alg_family(seed: int = _SEED + 2355) -> dict[str, float]:
    return _floats(_finite_blob("brauer_alg", bench_brauer_alg(seed)))


def bench_hecke_alg_family(seed: int = _SEED + 2356) -> dict[str, float]:
    return _floats(_finite_blob("hecke_alg", bench_hecke_alg(seed)))


def bench_casimir_op_family(seed: int = _SEED + 2357) -> dict[str, float]:
    return _floats(_finite_blob("casimir_op", bench_casimir_op(seed)))


def bench_weight_space_family(seed: int = _SEED + 2358) -> dict[str, float]:
    return _floats(_finite_blob("weight_space", bench_weight_space(seed)))


def bench_bz_category_family(seed: int = _SEED + 2359) -> dict[str, float]:
    return _floats(_finite_blob("bz_category", bench_bz_category(seed)))

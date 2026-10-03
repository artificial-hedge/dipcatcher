"""Wave-462 infinity-topos-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cartesian_fib2 import bench_cartesian_fib2
from quant_fund.models.cohesive_struct import bench_cohesive_struct
from quant_fund.models.descent_cond import bench_descent_cond
from quant_fund.models.lex_reflect import bench_lex_reflect
from quant_fund.models.n_localic import bench_n_localic
from quant_fund.models.shape_theory import bench_shape_theory

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


def bench_n_localic_family(seed: int = _SEED + 2678) -> dict[str, float]:
    return _floats(_finite_blob("n_localic", bench_n_localic(seed)))


def bench_shape_theory_family(seed: int = _SEED + 2679) -> dict[str, float]:
    return _floats(_finite_blob("shape_theory", bench_shape_theory(seed)))


def bench_descent_cond_family(seed: int = _SEED + 2680) -> dict[str, float]:
    return _floats(_finite_blob("descent_cond", bench_descent_cond(seed)))


def bench_lex_reflect_family(seed: int = _SEED + 2681) -> dict[str, float]:
    return _floats(_finite_blob("lex_reflect", bench_lex_reflect(seed)))


def bench_cartesian_fib2_family(seed: int = _SEED + 2682) -> dict[str, float]:
    return _floats(_finite_blob("cartesian_fib2", bench_cartesian_fib2(seed)))


def bench_cohesive_struct_family(seed: int = _SEED + 2683) -> dict[str, float]:
    return _floats(_finite_blob("cohesive_struct", bench_cohesive_struct(seed)))

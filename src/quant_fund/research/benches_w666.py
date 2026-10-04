"""Wave-666 derived-geometry-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cotangent_stack import bench_cotangent_stack
from quant_fund.models.derived_abelian import bench_derived_abelian
from quant_fund.models.derived_bezout import bench_derived_bezout
from quant_fund.models.derived_bun import bench_derived_bun
from quant_fund.models.derived_hecke import bench_derived_hecke
from quant_fund.models.simplicial_comm import bench_simplicial_comm

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


def bench_derived_abelian_family(
    seed: int = _SEED + 5500,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_abelian", bench_derived_abelian(seed)))


def bench_simplicial_comm_family(
    seed: int = _SEED + 5501,
) -> dict[str, float]:
    return _floats(_finite_blob("simplicial_comm", bench_simplicial_comm(seed)))


def bench_derived_bezout_family(
    seed: int = _SEED + 5502,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_bezout", bench_derived_bezout(seed)))


def bench_derived_hecke_family(
    seed: int = _SEED + 5503,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_hecke", bench_derived_hecke(seed)))


def bench_cotangent_stack_family(
    seed: int = _SEED + 5504,
) -> dict[str, float]:
    return _floats(_finite_blob("cotangent_stack", bench_cotangent_stack(seed)))


def bench_derived_bun_family(
    seed: int = _SEED + 5505,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_bun", bench_derived_bun(seed)))

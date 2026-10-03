"""Wave-459 order-theory-2/domain-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chain_cond import bench_chain_cond
from quant_fund.models.denotational import bench_denotational
from quant_fund.models.fixed_points_ord import bench_fixed_points_ord
from quant_fund.models.galois_insertion import bench_galois_insertion
from quant_fund.models.scott_cpo import bench_scott_cpo
from quant_fund.models.way_below import bench_way_below

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


def bench_fixed_points_ord_family(seed: int = _SEED + 2660) -> dict[str, float]:
    return _floats(_finite_blob("fixed_points_ord", bench_fixed_points_ord(seed)))


def bench_chain_cond_family(seed: int = _SEED + 2661) -> dict[str, float]:
    return _floats(_finite_blob("chain_cond", bench_chain_cond(seed)))


def bench_scott_cpo_family(seed: int = _SEED + 2662) -> dict[str, float]:
    return _floats(_finite_blob("scott_cpo", bench_scott_cpo(seed)))


def bench_way_below_family(seed: int = _SEED + 2663) -> dict[str, float]:
    return _floats(_finite_blob("way_below", bench_way_below(seed)))


def bench_galois_insertion_family(seed: int = _SEED + 2664) -> dict[str, float]:
    return _floats(_finite_blob("galois_insertion", bench_galois_insertion(seed)))


def bench_denotational_family(seed: int = _SEED + 2665) -> dict[str, float]:
    return _floats(_finite_blob("denotational", bench_denotational(seed)))

"""Wave-289 category-theory canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adjunction import bench_adjunction
from quant_fund.models.fin_cat import bench_fin_cat
from quant_fund.models.functor_check import bench_functor_check
from quant_fund.models.limit_prod import bench_limit_prod
from quant_fund.models.monad_laws import bench_monad_laws
from quant_fund.models.nat_trans import bench_nat_trans

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_fin_cat_family(seed: int = _SEED + 1640) -> dict[str, float]:
    return _floats(_finite_blob("fin_cat", bench_fin_cat(seed)))


def bench_functor_check_family(seed: int = _SEED + 1641) -> dict[str, float]:
    return _floats(_finite_blob("functor_check", bench_functor_check(seed)))


def bench_nat_trans_family(seed: int = _SEED + 1642) -> dict[str, float]:
    return _floats(_finite_blob("nat_trans", bench_nat_trans(seed)))


def bench_adjunction_family(seed: int = _SEED + 1643) -> dict[str, float]:
    return _floats(_finite_blob("adjunction", bench_adjunction(seed)))


def bench_limit_prod_family(seed: int = _SEED + 1644) -> dict[str, float]:
    return _floats(_finite_blob("limit_prod", bench_limit_prod(seed)))


def bench_monad_laws_family(seed: int = _SEED + 1645) -> dict[str, float]:
    return _floats(_finite_blob("monad_laws", bench_monad_laws(seed)))

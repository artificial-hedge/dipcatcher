"""Wave-342 descriptive-set-theory/recursion-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.analytic_sets import bench_analytic_sets
from quant_fund.models.arith_hierarchy import bench_arith_hierarchy
from quant_fund.models.borel_hierarchy import bench_borel_hierarchy
from quant_fund.models.forcing_lite import bench_forcing_lite
from quant_fund.models.jump_operator import bench_jump_operator
from quant_fund.models.rice_theorem import bench_rice_theorem

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


def bench_borel_hierarchy_family(seed: int = _SEED + 1959) -> dict[str, float]:
    return _floats(_finite_blob("borel_hierarchy", bench_borel_hierarchy(seed)))


def bench_analytic_sets_family(seed: int = _SEED + 1960) -> dict[str, float]:
    return _floats(_finite_blob("analytic_sets", bench_analytic_sets(seed)))


def bench_forcing_lite_family(seed: int = _SEED + 1961) -> dict[str, float]:
    return _floats(_finite_blob("forcing_lite", bench_forcing_lite(seed)))


def bench_arith_hierarchy_family(seed: int = _SEED + 1962) -> dict[str, float]:
    return _floats(_finite_blob("arith_hierarchy", bench_arith_hierarchy(seed)))


def bench_jump_operator_family(seed: int = _SEED + 1963) -> dict[str, float]:
    return _floats(_finite_blob("jump_operator", bench_jump_operator(seed)))


def bench_rice_theorem_family(seed: int = _SEED + 1964) -> dict[str, float]:
    return _floats(_finite_blob("rice_theorem", bench_rice_theorem(seed)))

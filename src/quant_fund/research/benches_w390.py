"""Wave-390 design-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.finite_difference import bench_finite_difference
from quant_fund.models.hadamard_matrix import bench_hadamard_matrix
from quant_fund.models.inc_structure import bench_inc_structure
from quant_fund.models.latin_trade import bench_latin_trade
from quant_fund.models.orthogonal_array import bench_orthogonal_array
from quant_fund.models.steiner_system import bench_steiner_system

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


def bench_latin_trade_family(seed: int = _SEED + 2246) -> dict[str, float]:
    return _floats(_finite_blob("latin_trade", bench_latin_trade(seed)))


def bench_steiner_system_family(seed: int = _SEED + 2247) -> dict[str, float]:
    return _floats(_finite_blob("steiner_system", bench_steiner_system(seed)))


def bench_inc_structure_family(seed: int = _SEED + 2248) -> dict[str, float]:
    return _floats(_finite_blob("inc_structure", bench_inc_structure(seed)))


def bench_orthogonal_array_family(seed: int = _SEED + 2249) -> dict[str, float]:
    return _floats(_finite_blob("orthogonal_array", bench_orthogonal_array(seed)))


def bench_hadamard_matrix_family(seed: int = _SEED + 2250) -> dict[str, float]:
    return _floats(_finite_blob("hadamard_matrix", bench_hadamard_matrix(seed)))


def bench_finite_difference_family(
    seed: int = _SEED + 2251,
) -> dict[str, float]:
    return _floats(_finite_blob("finite_difference", bench_finite_difference(seed)))

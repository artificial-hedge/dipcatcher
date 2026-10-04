"""Wave-398 representation-theory-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.artins_theorem import bench_artins_theorem
from quant_fund.models.clifford_toy import bench_clifford_toy
from quant_fund.models.frobenius_group import bench_frobenius_group
from quant_fund.models.induced_char import bench_induced_char
from quant_fund.models.schur_index import bench_schur_index
from quant_fund.models.tensor_char import bench_tensor_char

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


def bench_induced_char_family(seed: int = _SEED + 2294) -> dict[str, float]:
    return _floats(_finite_blob("induced_char", bench_induced_char(seed)))


def bench_artins_theorem_family(seed: int = _SEED + 2295) -> dict[str, float]:
    return _floats(_finite_blob("artins_theorem", bench_artins_theorem(seed)))


def bench_tensor_char_family(seed: int = _SEED + 2296) -> dict[str, float]:
    return _floats(_finite_blob("tensor_char", bench_tensor_char(seed)))


def bench_clifford_toy_family(seed: int = _SEED + 2297) -> dict[str, float]:
    return _floats(_finite_blob("clifford_toy", bench_clifford_toy(seed)))


def bench_schur_index_family(seed: int = _SEED + 2298) -> dict[str, float]:
    return _floats(_finite_blob("schur_index", bench_schur_index(seed)))


def bench_frobenius_group_family(
    seed: int = _SEED + 2299,
) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_group", bench_frobenius_group(seed)))

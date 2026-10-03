"""Wave-343 lattice/universal-algebra canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boolean_algebra import bench_boolean_algebra
from quant_fund.models.congruence_lattice import bench_congruence_lattice
from quant_fund.models.galois_connection import bench_galois_connection
from quant_fund.models.lattice_check import bench_lattice_check
from quant_fund.models.tarski_fixed import bench_tarski_fixed
from quant_fund.models.term_algebra import bench_term_algebra

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


def bench_lattice_check_family(seed: int = _SEED + 1965) -> dict[str, float]:
    return _floats(_finite_blob("lattice_check", bench_lattice_check(seed)))


def bench_galois_connection_family(seed: int = _SEED + 1966) -> dict[str, float]:
    return _floats(_finite_blob("galois_connection", bench_galois_connection(seed)))


def bench_tarski_fixed_family(seed: int = _SEED + 1967) -> dict[str, float]:
    return _floats(_finite_blob("tarski_fixed", bench_tarski_fixed(seed)))


def bench_boolean_algebra_family(seed: int = _SEED + 1968) -> dict[str, float]:
    return _floats(_finite_blob("boolean_algebra", bench_boolean_algebra(seed)))


def bench_congruence_lattice_family(seed: int = _SEED + 1969) -> dict[str, float]:
    return _floats(_finite_blob("congruence_lattice", bench_congruence_lattice(seed)))


def bench_term_algebra_family(seed: int = _SEED + 1970) -> dict[str, float]:
    return _floats(_finite_blob("term_algebra", bench_term_algebra(seed)))

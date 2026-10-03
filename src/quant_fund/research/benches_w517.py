"""Wave-517 quantum-groups bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.crystal_base import bench_crystal_base
from quant_fund.models.jimbo_drin import bench_jimbo_drin
from quant_fund.models.lusztig_can import bench_lusztig_can
from quant_fund.models.quantum_group import bench_quantum_group
from quant_fund.models.quantum_rmatrix import bench_quantum_rmatrix
from quant_fund.models.quantum_schur import bench_quantum_schur

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


def bench_quantum_group_family(seed: int = _SEED + 3008) -> dict[str, float]:
    return _floats(_finite_blob("quantum_group", bench_quantum_group(seed)))


def bench_crystal_base_family(seed: int = _SEED + 3009) -> dict[str, float]:
    return _floats(_finite_blob("crystal_base", bench_crystal_base(seed)))


def bench_quantum_rmatrix_family(seed: int = _SEED + 3010) -> dict[str, float]:
    return _floats(_finite_blob("quantum_rmatrix", bench_quantum_rmatrix(seed)))


def bench_jimbo_drin_family(seed: int = _SEED + 3011) -> dict[str, float]:
    return _floats(_finite_blob("jimbo_drin", bench_jimbo_drin(seed)))


def bench_lusztig_can_family(seed: int = _SEED + 3012) -> dict[str, float]:
    return _floats(_finite_blob("lusztig_can", bench_lusztig_can(seed)))


def bench_quantum_schur_family(seed: int = _SEED + 3013) -> dict[str, float]:
    return _floats(_finite_blob("quantum_schur", bench_quantum_schur(seed)))

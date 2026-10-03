"""Wave-310 quantum-error-correction canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gottesman_knill import bench_gottesman_knill
from quant_fund.models.repetition_qec import bench_repetition_qec
from quant_fund.models.shor_code import bench_shor_code
from quant_fund.models.steane_code import bench_steane_code
from quant_fund.models.surface_code import bench_surface_code
from quant_fund.models.syndrome_circuit import bench_syndrome_circuit

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


def bench_gottesman_knill_family(seed: int = _SEED + 1766) -> dict[str, float]:
    return _floats(_finite_blob("gottesman_knill", bench_gottesman_knill(seed)))


def bench_steane_code_family(seed: int = _SEED + 1767) -> dict[str, float]:
    return _floats(_finite_blob("steane_code", bench_steane_code(seed)))


def bench_surface_code_family(seed: int = _SEED + 1768) -> dict[str, float]:
    return _floats(_finite_blob("surface_code", bench_surface_code(seed)))


def bench_shor_code_family(seed: int = _SEED + 1769) -> dict[str, float]:
    return _floats(_finite_blob("shor_code", bench_shor_code(seed)))


def bench_syndrome_circuit_family(seed: int = _SEED + 1770) -> dict[str, float]:
    return _floats(_finite_blob("syndrome_circuit", bench_syndrome_circuit(seed)))


def bench_repetition_qec_family(seed: int = _SEED + 1771) -> dict[str, float]:
    return _floats(_finite_blob("repetition_qec", bench_repetition_qec(seed)))

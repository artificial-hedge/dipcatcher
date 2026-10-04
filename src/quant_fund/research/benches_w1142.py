"""Wave-1142 quantum-technology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.quantum_chemistry_2 import bench_quantum_chemistry_2
from quant_fund.models.quantum_computing import bench_quantum_computing
from quant_fund.models.quantum_error_2 import bench_quantum_error_2
from quant_fund.models.quantum_information_2 import bench_quantum_information_2
from quant_fund.models.quantum_optics import bench_quantum_optics
from quant_fund.models.quantum_sensing import bench_quantum_sensing

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_quantum_computing_family(seed: int = _SEED + 53000) -> dict[str, float]:
    return _finite_blob(bench_quantum_computing(seed))


def bench_quantum_information_2_family(seed: int = _SEED + 53001) -> dict[str, float]:
    return _finite_blob(bench_quantum_information_2(seed))


def bench_quantum_chemistry_2_family(seed: int = _SEED + 53002) -> dict[str, float]:
    return _finite_blob(bench_quantum_chemistry_2(seed))


def bench_quantum_optics_family(seed: int = _SEED + 53003) -> dict[str, float]:
    return _finite_blob(bench_quantum_optics(seed))


def bench_quantum_sensing_family(seed: int = _SEED + 53004) -> dict[str, float]:
    return _finite_blob(bench_quantum_sensing(seed))


def bench_quantum_error_2_family(seed: int = _SEED + 53005) -> dict[str, float]:
    return _finite_blob(bench_quantum_error_2(seed))

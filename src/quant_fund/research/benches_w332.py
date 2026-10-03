"""Wave-332 quantum-information canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bell_ineq import bench_bell_ineq
from quant_fund.models.density_matrix import bench_density_matrix
from quant_fund.models.entanglement import bench_entanglement
from quant_fund.models.povm_measure import bench_povm_measure
from quant_fund.models.qchannel import bench_qchannel
from quant_fund.models.state_tomo import bench_state_tomo

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


def bench_density_matrix_family(seed: int = _SEED + 1899) -> dict[str, float]:
    return _floats(_finite_blob("density_matrix", bench_density_matrix(seed)))


def bench_povm_measure_family(seed: int = _SEED + 1900) -> dict[str, float]:
    return _floats(_finite_blob("povm_measure", bench_povm_measure(seed)))


def bench_qchannel_family(seed: int = _SEED + 1901) -> dict[str, float]:
    return _floats(_finite_blob("qchannel", bench_qchannel(seed)))


def bench_entanglement_family(seed: int = _SEED + 1902) -> dict[str, float]:
    return _floats(_finite_blob("entanglement", bench_entanglement(seed)))


def bench_bell_ineq_family(seed: int = _SEED + 1903) -> dict[str, float]:
    return _floats(_finite_blob("bell_ineq", bench_bell_ineq(seed)))


def bench_state_tomo_family(seed: int = _SEED + 1904) -> dict[str, float]:
    return _floats(_finite_blob("state_tomo", bench_state_tomo(seed)))

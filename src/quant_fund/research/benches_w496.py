"""Wave-496 DAG-deformation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dag_representation import bench_dag_representation
from quant_fund.models.derived_deformation import bench_derived_deformation
from quant_fund.models.derived_moduli import bench_derived_moduli
from quant_fund.models.formal_deformation import bench_formal_deformation
from quant_fund.models.obstruction_2 import bench_obstruction_2
from quant_fund.models.tangent_coh import bench_tangent_coh

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


def bench_derived_deformation_family(seed: int = _SEED + 2882) -> dict[str, float]:
    return _floats(_finite_blob("derived_deformation", bench_derived_deformation(seed)))


def bench_formal_deformation_family(seed: int = _SEED + 2883) -> dict[str, float]:
    return _floats(_finite_blob("formal_deformation", bench_formal_deformation(seed)))


def bench_dag_representation_family(seed: int = _SEED + 2884) -> dict[str, float]:
    return _floats(_finite_blob("dag_representation", bench_dag_representation(seed)))


def bench_derived_moduli_family(seed: int = _SEED + 2885) -> dict[str, float]:
    return _floats(_finite_blob("derived_moduli", bench_derived_moduli(seed)))


def bench_tangent_coh_family(seed: int = _SEED + 2886) -> dict[str, float]:
    return _floats(_finite_blob("tangent_coh", bench_tangent_coh(seed)))


def bench_obstruction_2_family(seed: int = _SEED + 2887) -> dict[str, float]:
    return _floats(_finite_blob("obstruction_2", bench_obstruction_2(seed)))

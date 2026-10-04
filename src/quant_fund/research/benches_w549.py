"""Wave-549 index-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.analytic_torsion import bench_analytic_torsion
from quant_fund.models.atiyah_singer import bench_atiyah_singer
from quant_fund.models.dirac_op import bench_dirac_op
from quant_fund.models.eta_invariant import bench_eta_invariant
from quant_fund.models.heat_kernel2 import bench_heat_kernel2
from quant_fund.models.signature_op import bench_signature_op

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


def bench_atiyah_singer_family(seed: int = _SEED + 3200) -> dict[str, float]:
    return _floats(_finite_blob("atiyah_singer", bench_atiyah_singer(seed)))


def bench_dirac_op_family(seed: int = _SEED + 3201) -> dict[str, float]:
    return _floats(_finite_blob("dirac_op", bench_dirac_op(seed)))


def bench_eta_invariant_family(seed: int = _SEED + 3202) -> dict[str, float]:
    return _floats(_finite_blob("eta_invariant", bench_eta_invariant(seed)))


def bench_heat_kernel2_family(seed: int = _SEED + 3203) -> dict[str, float]:
    return _floats(_finite_blob("heat_kernel2", bench_heat_kernel2(seed)))


def bench_signature_op_family(seed: int = _SEED + 3204) -> dict[str, float]:
    return _floats(_finite_blob("signature_op", bench_signature_op(seed)))


def bench_analytic_torsion_family(
    seed: int = _SEED + 3205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "analytic_torsion",
            bench_analytic_torsion(seed),
        )
    )

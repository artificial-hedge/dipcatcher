"""Wave-387 probability-4 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.concentration_ineq import bench_concentration_ineq
from quant_fund.models.kolmogorov_01 import bench_kolmogorov_01
from quant_fund.models.ldp_theory import bench_ldp_theory
from quant_fund.models.prokhorov_metric import bench_prokhorov_metric
from quant_fund.models.uniform_integrability import bench_uniform_integrability
from quant_fund.models.vitali_conv import bench_vitali_conv

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


def bench_uniform_integrability_family(
    seed: int = _SEED + 2228,
) -> dict[str, float]:
    return _floats(_finite_blob("uniform_integrability", bench_uniform_integrability(seed)))


def bench_vitali_conv_family(seed: int = _SEED + 2229) -> dict[str, float]:
    return _floats(_finite_blob("vitali_conv", bench_vitali_conv(seed)))


def bench_ldp_theory_family(seed: int = _SEED + 2230) -> dict[str, float]:
    return _floats(_finite_blob("ldp_theory", bench_ldp_theory(seed)))


def bench_concentration_ineq_family(
    seed: int = _SEED + 2231,
) -> dict[str, float]:
    return _floats(_finite_blob("concentration_ineq", bench_concentration_ineq(seed)))


def bench_kolmogorov_01_family(seed: int = _SEED + 2232) -> dict[str, float]:
    return _floats(_finite_blob("kolmogorov_01", bench_kolmogorov_01(seed)))


def bench_prokhorov_metric_family(seed: int = _SEED + 2233) -> dict[str, float]:
    return _floats(_finite_blob("prokhorov_metric", bench_prokhorov_metric(seed)))

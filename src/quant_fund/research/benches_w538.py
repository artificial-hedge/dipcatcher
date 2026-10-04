"""Wave-538 elliptic-PDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.degiorgi_nash import bench_degiorgi_nash
from quant_fund.models.harnack_thm import bench_harnack_thm
from quant_fund.models.poincare_ineq import bench_poincare_ineq
from quant_fund.models.schauder_est import bench_schauder_est
from quant_fund.models.sobolev_space import bench_sobolev_space
from quant_fund.models.trace_thm import bench_trace_thm

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


def bench_sobolev_space_family(seed: int = _SEED + 3134) -> dict[str, float]:
    return _floats(_finite_blob("sobolev_space", bench_sobolev_space(seed)))


def bench_poincare_ineq_family(seed: int = _SEED + 3135) -> dict[str, float]:
    return _floats(_finite_blob("poincare_ineq", bench_poincare_ineq(seed)))


def bench_trace_thm_family(seed: int = _SEED + 3136) -> dict[str, float]:
    return _floats(_finite_blob("trace_thm", bench_trace_thm(seed)))


def bench_harnack_thm_family(seed: int = _SEED + 3137) -> dict[str, float]:
    return _floats(_finite_blob("harnack_thm", bench_harnack_thm(seed)))


def bench_schauder_est_family(seed: int = _SEED + 3138) -> dict[str, float]:
    return _floats(_finite_blob("schauder_est", bench_schauder_est(seed)))


def bench_degiorgi_nash_family(
    seed: int = _SEED + 3139,
) -> dict[str, float]:
    return _floats(_finite_blob("degiorgi_nash", bench_degiorgi_nash(seed)))

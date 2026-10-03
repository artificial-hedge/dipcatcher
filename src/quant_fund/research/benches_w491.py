"""Wave-491 automorphic-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arthur_param import bench_arthur_param
from quant_fund.models.hecke_alg2 import bench_hecke_alg2
from quant_fund.models.l_function import bench_l_function
from quant_fund.models.satake_param import bench_satake_param
from quant_fund.models.shimura_var import bench_shimura_var
from quant_fund.models.theta_lift import bench_theta_lift

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


def bench_shimura_var_family(seed: int = _SEED + 2852) -> dict[str, float]:
    return _floats(_finite_blob("shimura_var", bench_shimura_var(seed)))


def bench_l_function_family(seed: int = _SEED + 2853) -> dict[str, float]:
    return _floats(_finite_blob("l_function", bench_l_function(seed)))


def bench_hecke_alg2_family(seed: int = _SEED + 2854) -> dict[str, float]:
    return _floats(_finite_blob("hecke_alg2", bench_hecke_alg2(seed)))


def bench_theta_lift_family(seed: int = _SEED + 2855) -> dict[str, float]:
    return _floats(_finite_blob("theta_lift", bench_theta_lift(seed)))


def bench_arthur_param_family(seed: int = _SEED + 2856) -> dict[str, float]:
    return _floats(_finite_blob("arthur_param", bench_arthur_param(seed)))


def bench_satake_param_family(seed: int = _SEED + 2857) -> dict[str, float]:
    return _floats(_finite_blob("satake_param", bench_satake_param(seed)))

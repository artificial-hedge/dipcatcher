"""Wave-446 Goodwillie-calculus bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.calc_converge import bench_calc_converge
from quant_fund.models.deriv_layer import bench_deriv_layer
from quant_fund.models.excisive_fn import bench_excisive_fn
from quant_fund.models.goodwillie_tower import bench_goodwillie_tower
from quant_fund.models.linearization import bench_linearization
from quant_fund.models.orth_calc import bench_orth_calc

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


def bench_goodwillie_tower_family(seed: int = _SEED + 2582) -> dict[str, float]:
    return _floats(_finite_blob("goodwillie_tower", bench_goodwillie_tower(seed)))


def bench_excisive_fn_family(seed: int = _SEED + 2583) -> dict[str, float]:
    return _floats(_finite_blob("excisive_fn", bench_excisive_fn(seed)))


def bench_linearization_family(seed: int = _SEED + 2584) -> dict[str, float]:
    return _floats(_finite_blob("linearization", bench_linearization(seed)))


def bench_deriv_layer_family(seed: int = _SEED + 2585) -> dict[str, float]:
    return _floats(_finite_blob("deriv_layer", bench_deriv_layer(seed)))


def bench_calc_converge_family(seed: int = _SEED + 2586) -> dict[str, float]:
    return _floats(_finite_blob("calc_converge", bench_calc_converge(seed)))


def bench_orth_calc_family(seed: int = _SEED + 2587) -> dict[str, float]:
    return _floats(_finite_blob("orth_calc", bench_orth_calc(seed)))

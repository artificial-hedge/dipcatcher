"""Wave-338 set-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ac_choice import bench_ac_choice
from quant_fund.models.cardinal_arith import bench_cardinal_arith
from quant_fund.models.ordinal_arith import bench_ordinal_arith
from quant_fund.models.transfinite_induct import bench_transfinite_induct
from quant_fund.models.v_omega import bench_v_omega
from quant_fund.models.well_founded import bench_well_founded

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


def bench_ordinal_arith_family(seed: int = _SEED + 1935) -> dict[str, float]:
    return _floats(_finite_blob("ordinal_arith", bench_ordinal_arith(seed)))


def bench_cardinal_arith_family(seed: int = _SEED + 1936) -> dict[str, float]:
    return _floats(_finite_blob("cardinal_arith", bench_cardinal_arith(seed)))


def bench_transfinite_induct_family(seed: int = _SEED + 1937) -> dict[str, float]:
    return _floats(_finite_blob("transfinite_induct", bench_transfinite_induct(seed)))


def bench_well_founded_family(seed: int = _SEED + 1938) -> dict[str, float]:
    return _floats(_finite_blob("well_founded", bench_well_founded(seed)))


def bench_v_omega_family(seed: int = _SEED + 1939) -> dict[str, float]:
    return _floats(_finite_blob("v_omega", bench_v_omega(seed)))


def bench_ac_choice_family(seed: int = _SEED + 1940) -> dict[str, float]:
    return _floats(_finite_blob("ac_choice", bench_ac_choice(seed)))

"""Wave-401 proof-theory-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cut_elim_seq import bench_cut_elim_seq
from quant_fund.models.finitary_induct import bench_finitary_induct
from quant_fund.models.herbrand_thm import bench_herbrand_thm
from quant_fund.models.hilbert_system import bench_hilbert_system
from quant_fund.models.interp_equality import bench_interp_equality
from quant_fund.models.reverse_math import bench_reverse_math

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


def bench_herbrand_thm_family(seed: int = _SEED + 2312) -> dict[str, float]:
    return _floats(_finite_blob("herbrand_thm", bench_herbrand_thm(seed)))


def bench_interp_equality_family(
    seed: int = _SEED + 2313,
) -> dict[str, float]:
    return _floats(_finite_blob("interp_equality", bench_interp_equality(seed)))


def bench_cut_elim_seq_family(seed: int = _SEED + 2314) -> dict[str, float]:
    return _floats(_finite_blob("cut_elim_seq", bench_cut_elim_seq(seed)))


def bench_finitary_induct_family(
    seed: int = _SEED + 2315,
) -> dict[str, float]:
    return _floats(_finite_blob("finitary_induct", bench_finitary_induct(seed)))


def bench_hilbert_system_family(
    seed: int = _SEED + 2316,
) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_system", bench_hilbert_system(seed)))


def bench_reverse_math_family(seed: int = _SEED + 2317) -> dict[str, float]:
    return _floats(_finite_blob("reverse_math", bench_reverse_math(seed)))

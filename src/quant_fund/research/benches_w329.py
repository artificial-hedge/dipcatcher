"""Wave-329 proof-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cut_elim import bench_cut_elim
from quant_fund.models.intuit_class import bench_intuit_class
from quant_fund.models.linear_logic import bench_linear_logic
from quant_fund.models.nd_check import bench_nd_check
from quant_fund.models.resolution_fol import bench_resolution_fol
from quant_fund.models.sequent_prove import bench_sequent_prove

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


def bench_nd_check_family(seed: int = _SEED + 1881) -> dict[str, float]:
    return _floats(_finite_blob("nd_check", bench_nd_check(seed)))


def bench_sequent_prove_family(seed: int = _SEED + 1882) -> dict[str, float]:
    return _floats(_finite_blob("sequent_prove", bench_sequent_prove(seed)))


def bench_cut_elim_family(seed: int = _SEED + 1883) -> dict[str, float]:
    return _floats(_finite_blob("cut_elim", bench_cut_elim(seed)))


def bench_resolution_fol_family(seed: int = _SEED + 1884) -> dict[str, float]:
    return _floats(_finite_blob("resolution_fol", bench_resolution_fol(seed)))


def bench_linear_logic_family(seed: int = _SEED + 1885) -> dict[str, float]:
    return _floats(_finite_blob("linear_logic", bench_linear_logic(seed)))


def bench_intuit_class_family(seed: int = _SEED + 1886) -> dict[str, float]:
    return _floats(_finite_blob("intuit_class", bench_intuit_class(seed)))

"""Wave-322 verification-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cegis_loop import bench_cegis_loop
from quant_fund.models.horn_clauses import bench_horn_clauses
from quant_fund.models.interpolant_mc import bench_interpolant_mc
from quant_fund.models.predicate_abs import bench_predicate_abs
from quant_fund.models.sygus_synth import bench_sygus_synth
from quant_fund.models.weakest_precond import bench_weakest_precond

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


def bench_weakest_precond_family(seed: int = _SEED + 1839) -> dict[str, float]:
    return _floats(_finite_blob("weakest_precond", bench_weakest_precond(seed)))


def bench_sygus_synth_family(seed: int = _SEED + 1840) -> dict[str, float]:
    return _floats(_finite_blob("sygus_synth", bench_sygus_synth(seed)))


def bench_horn_clauses_family(seed: int = _SEED + 1841) -> dict[str, float]:
    return _floats(_finite_blob("horn_clauses", bench_horn_clauses(seed)))


def bench_interpolant_mc_family(seed: int = _SEED + 1842) -> dict[str, float]:
    return _floats(_finite_blob("interpolant_mc", bench_interpolant_mc(seed)))


def bench_predicate_abs_family(seed: int = _SEED + 1843) -> dict[str, float]:
    return _floats(_finite_blob("predicate_abs", bench_predicate_abs(seed)))


def bench_cegis_loop_family(seed: int = _SEED + 1844) -> dict[str, float]:
    return _floats(_finite_blob("cegis_loop", bench_cegis_loop(seed)))

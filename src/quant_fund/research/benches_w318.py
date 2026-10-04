"""Wave-318 type-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bidirectional_tc import bench_bidirectional_tc
from quant_fund.models.dep_types import bench_dep_types
from quant_fund.models.nbe_eval import bench_nbe_eval
from quant_fund.models.proof_kernel import bench_proof_kernel
from quant_fund.models.tactic_engine import bench_tactic_engine
from quant_fund.models.unify_meta import bench_unify_meta

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


def bench_bidirectional_tc_family(seed: int = _SEED + 1815) -> dict[str, float]:
    return _floats(_finite_blob("bidirectional_tc", bench_bidirectional_tc(seed)))


def bench_nbe_eval_family(seed: int = _SEED + 1816) -> dict[str, float]:
    return _floats(_finite_blob("nbe_eval", bench_nbe_eval(seed)))


def bench_dep_types_family(seed: int = _SEED + 1817) -> dict[str, float]:
    return _floats(_finite_blob("dep_types", bench_dep_types(seed)))


def bench_unify_meta_family(seed: int = _SEED + 1818) -> dict[str, float]:
    return _floats(_finite_blob("unify_meta", bench_unify_meta(seed)))


def bench_proof_kernel_family(seed: int = _SEED + 1819) -> dict[str, float]:
    return _floats(_finite_blob("proof_kernel", bench_proof_kernel(seed)))


def bench_tactic_engine_family(seed: int = _SEED + 1820) -> dict[str, float]:
    return _floats(_finite_blob("tactic_engine", bench_tactic_engine(seed)))

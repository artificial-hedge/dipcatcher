"""Wave-386 proof-theory-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.godel_incomp import bench_godel_incomp
from quant_fund.models.interp_proof import bench_interp_proof
from quant_fund.models.modal_completeness import bench_modal_completeness
from quant_fund.models.natural_ded import bench_natural_ded
from quant_fund.models.proof_complexity import bench_proof_complexity
from quant_fund.models.sequent_calculus import bench_sequent_calculus

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


def bench_sequent_calculus_family(seed: int = _SEED + 2222) -> dict[str, float]:
    return _floats(_finite_blob("sequent_calculus", bench_sequent_calculus(seed)))


def bench_natural_ded_family(seed: int = _SEED + 2223) -> dict[str, float]:
    return _floats(_finite_blob("natural_ded", bench_natural_ded(seed)))


def bench_godel_incomp_family(seed: int = _SEED + 2224) -> dict[str, float]:
    return _floats(_finite_blob("godel_incomp", bench_godel_incomp(seed)))


def bench_interp_proof_family(seed: int = _SEED + 2225) -> dict[str, float]:
    return _floats(_finite_blob("interp_proof", bench_interp_proof(seed)))


def bench_proof_complexity_family(seed: int = _SEED + 2226) -> dict[str, float]:
    return _floats(_finite_blob("proof_complexity", bench_proof_complexity(seed)))


def bench_modal_completeness_family(
    seed: int = _SEED + 2227,
) -> dict[str, float]:
    return _floats(_finite_blob("modal_completeness", bench_modal_completeness(seed)))

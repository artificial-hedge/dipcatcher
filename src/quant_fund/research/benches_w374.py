"""Wave-374 model-theory-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.acl_closure import bench_acl_closure
from quant_fund.models.morley_rank import bench_morley_rank
from quant_fund.models.omega_categoricity import bench_omega_categoricity
from quant_fund.models.quantifier_elim import bench_quantifier_elim
from quant_fund.models.realize_types import bench_realize_types
from quant_fund.models.vocab_interp import bench_vocab_interp

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


def bench_quantifier_elim_family(seed: int = _SEED + 2150) -> dict[str, float]:
    return _floats(_finite_blob("quantifier_elim", bench_quantifier_elim(seed)))


def bench_realize_types_family(seed: int = _SEED + 2151) -> dict[str, float]:
    return _floats(_finite_blob("realize_types", bench_realize_types(seed)))


def bench_omega_categoricity_family(seed: int = _SEED + 2152) -> dict[str, float]:
    return _floats(_finite_blob("omega_categoricity", bench_omega_categoricity(seed)))


def bench_acl_closure_family(seed: int = _SEED + 2153) -> dict[str, float]:
    return _floats(_finite_blob("acl_closure", bench_acl_closure(seed)))


def bench_morley_rank_family(seed: int = _SEED + 2154) -> dict[str, float]:
    return _floats(_finite_blob("morley_rank", bench_morley_rank(seed)))


def bench_vocab_interp_family(seed: int = _SEED + 2155) -> dict[str, float]:
    return _floats(_finite_blob("vocab_interp", bench_vocab_interp(seed)))

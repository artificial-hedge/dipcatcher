"""Wave-368 model-theory-2/logic canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.herbrand_model import bench_herbrand_model
from quant_fund.models.los_theorem import bench_los_theorem
from quant_fund.models.presburger import bench_presburger
from quant_fund.models.skolem_normal import bench_skolem_normal
from quant_fund.models.unification_fol import bench_unification_fol

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


def bench_unification_fol_family(seed: int = _SEED + 2115) -> dict[str, float]:
    return _floats(_finite_blob("unification_fol", bench_unification_fol(seed)))


def bench_skolem_normal_family(seed: int = _SEED + 2116) -> dict[str, float]:
    return _floats(_finite_blob("skolem_normal", bench_skolem_normal(seed)))


def bench_herbrand_model_family(seed: int = _SEED + 2117) -> dict[str, float]:
    return _floats(_finite_blob("herbrand_model", bench_herbrand_model(seed)))


def bench_presburger_family(seed: int = _SEED + 2118) -> dict[str, float]:
    return _floats(_finite_blob("presburger", bench_presburger(seed)))


def bench_los_theorem_family(seed: int = _SEED + 2119) -> dict[str, float]:
    return _floats(_finite_blob("los_theorem", bench_los_theorem(seed)))

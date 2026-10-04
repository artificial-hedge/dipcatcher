"""Wave-448 higher-topos bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cohesive_top import bench_cohesive_top
from quant_fund.models.hypercomplete import bench_hypercomplete
from quant_fund.models.infty_topos import bench_infty_topos
from quant_fund.models.object_classif import bench_object_classif
from quant_fund.models.trunc_modal import bench_trunc_modal
from quant_fund.models.univ_colimit import bench_univ_colimit

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


def bench_infty_topos_family(seed: int = _SEED + 2594) -> dict[str, float]:
    return _floats(_finite_blob("infty_topos", bench_infty_topos(seed)))


def bench_univ_colimit_family(seed: int = _SEED + 2595) -> dict[str, float]:
    return _floats(_finite_blob("univ_colimit", bench_univ_colimit(seed)))


def bench_object_classif_family(seed: int = _SEED + 2596) -> dict[str, float]:
    return _floats(_finite_blob("object_classif", bench_object_classif(seed)))


def bench_trunc_modal_family(seed: int = _SEED + 2597) -> dict[str, float]:
    return _floats(_finite_blob("trunc_modal", bench_trunc_modal(seed)))


def bench_cohesive_top_family(seed: int = _SEED + 2598) -> dict[str, float]:
    return _floats(_finite_blob("cohesive_top", bench_cohesive_top(seed)))


def bench_hypercomplete_family(seed: int = _SEED + 2599) -> dict[str, float]:
    return _floats(_finite_blob("hypercomplete", bench_hypercomplete(seed)))

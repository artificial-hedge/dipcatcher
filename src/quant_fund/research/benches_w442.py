"""Wave-442 model-categories-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cofibrant_rep import bench_cofibrant_rep
from quant_fund.models.enriched_model import bench_enriched_model
from quant_fund.models.localization_mc import (
    bench_localization_mc,
)
from quant_fund.models.monoidal_model import bench_monoidal_model
from quant_fund.models.quillen_equiv import bench_quillen_equiv
from quant_fund.models.reedy_model import bench_reedy_model

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


def bench_cofibrant_rep_family(
    seed: int = _SEED + 2558,
) -> dict[str, float]:
    return _floats(_finite_blob("cofibrant_rep", bench_cofibrant_rep(seed)))


def bench_quillen_equiv_family(
    seed: int = _SEED + 2559,
) -> dict[str, float]:
    return _floats(_finite_blob("quillen_equiv", bench_quillen_equiv(seed)))


def bench_monoidal_model_family(
    seed: int = _SEED + 2560,
) -> dict[str, float]:
    return _floats(_finite_blob("monoidal_model", bench_monoidal_model(seed)))


def bench_enriched_model_family(
    seed: int = _SEED + 2561,
) -> dict[str, float]:
    return _floats(_finite_blob("enriched_model", bench_enriched_model(seed)))


def bench_reedy_model_family(
    seed: int = _SEED + 2562,
) -> dict[str, float]:
    return _floats(_finite_blob("reedy_model", bench_reedy_model(seed)))


def bench_localization_mc_family(
    seed: int = _SEED + 2563,
) -> dict[str, float]:
    return _floats(_finite_blob("localization_mc", bench_localization_mc(seed)))

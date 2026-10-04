"""Wave-507 quasi-category/Joyal bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_coherent import bench_homotopy_coherent
from quant_fund.models.htc_colimit import bench_htc_colimit
from quant_fund.models.joyal_model import bench_joyal_model
from quant_fund.models.marking_qcat import bench_marking_qcat
from quant_fund.models.nerve_quasi import bench_nerve_quasi
from quant_fund.models.quasi_cat import bench_quasi_cat

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


def bench_quasi_cat_family(seed: int = _SEED + 2948) -> dict[str, float]:
    return _floats(_finite_blob("quasi_cat", bench_quasi_cat(seed)))


def bench_joyal_model_family(seed: int = _SEED + 2949) -> dict[str, float]:
    return _floats(_finite_blob("joyal_model", bench_joyal_model(seed)))


def bench_homotopy_coherent_family(seed: int = _SEED + 2950) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_coherent", bench_homotopy_coherent(seed)))


def bench_nerve_quasi_family(seed: int = _SEED + 2951) -> dict[str, float]:
    return _floats(_finite_blob("nerve_quasi", bench_nerve_quasi(seed)))


def bench_htc_colimit_family(seed: int = _SEED + 2952) -> dict[str, float]:
    return _floats(_finite_blob("htc_colimit", bench_htc_colimit(seed)))


def bench_marking_qcat_family(seed: int = _SEED + 2953) -> dict[str, float]:
    return _floats(_finite_blob("marking_qcat", bench_marking_qcat(seed)))

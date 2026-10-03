"""Wave-492 log-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.kato_fontaine import bench_kato_fontaine
from quant_fund.models.log_crystalline import bench_log_crystalline
from quant_fund.models.log_derham import bench_log_derham
from quant_fund.models.log_etale import bench_log_etale
from quant_fund.models.log_smooth import bench_log_smooth
from quant_fund.models.log_structure import bench_log_structure

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


def bench_log_structure_family(seed: int = _SEED + 2858) -> dict[str, float]:
    return _floats(_finite_blob("log_structure", bench_log_structure(seed)))


def bench_kato_fontaine_family(seed: int = _SEED + 2859) -> dict[str, float]:
    return _floats(_finite_blob("kato_fontaine", bench_kato_fontaine(seed)))


def bench_log_smooth_family(seed: int = _SEED + 2860) -> dict[str, float]:
    return _floats(_finite_blob("log_smooth", bench_log_smooth(seed)))


def bench_log_etale_family(seed: int = _SEED + 2861) -> dict[str, float]:
    return _floats(_finite_blob("log_etale", bench_log_etale(seed)))


def bench_log_derham_family(seed: int = _SEED + 2862) -> dict[str, float]:
    return _floats(_finite_blob("log_derham", bench_log_derham(seed)))


def bench_log_crystalline_family(seed: int = _SEED + 2863) -> dict[str, float]:
    return _floats(_finite_blob("log_crystalline", bench_log_crystalline(seed)))

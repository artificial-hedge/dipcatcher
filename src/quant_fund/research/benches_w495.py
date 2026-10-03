"""Wave-495 infinity-2-category bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.complicial import bench_complicial
from quant_fund.models.globular_model import bench_globular_model
from quant_fund.models.opetopic import bench_opetopic
from quant_fund.models.theta_space import bench_theta_space
from quant_fund.models.verity_gray import bench_verity_gray
from quant_fund.models.weak_infty import bench_weak_infty

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


def bench_globular_model_family(seed: int = _SEED + 2876) -> dict[str, float]:
    return _floats(_finite_blob("globular_model", bench_globular_model(seed)))


def bench_opetopic_family(seed: int = _SEED + 2877) -> dict[str, float]:
    return _floats(_finite_blob("opetopic", bench_opetopic(seed)))


def bench_theta_space_family(seed: int = _SEED + 2878) -> dict[str, float]:
    return _floats(_finite_blob("theta_space", bench_theta_space(seed)))


def bench_complicial_family(seed: int = _SEED + 2879) -> dict[str, float]:
    return _floats(_finite_blob("complicial", bench_complicial(seed)))


def bench_verity_gray_family(seed: int = _SEED + 2880) -> dict[str, float]:
    return _floats(_finite_blob("verity_gray", bench_verity_gray(seed)))


def bench_weak_infty_family(seed: int = _SEED + 2881) -> dict[str, float]:
    return _floats(_finite_blob("weak_infty", bench_weak_infty(seed)))

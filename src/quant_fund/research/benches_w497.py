"""Wave-497 birational-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fano_mori import bench_fano_mori
from quant_fund.models.flip_cone import bench_flip_cone
from quant_fund.models.klt_pair import bench_klt_pair
from quant_fund.models.minimal_model import bench_minimal_model
from quant_fund.models.mmp_algorithm import bench_mmp_algorithm
from quant_fund.models.toric_flip import bench_toric_flip

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


def bench_minimal_model_family(seed: int = _SEED + 2888) -> dict[str, float]:
    return _floats(_finite_blob("minimal_model", bench_minimal_model(seed)))


def bench_klt_pair_family(seed: int = _SEED + 2889) -> dict[str, float]:
    return _floats(_finite_blob("klt_pair", bench_klt_pair(seed)))


def bench_flip_cone_family(seed: int = _SEED + 2890) -> dict[str, float]:
    return _floats(_finite_blob("flip_cone", bench_flip_cone(seed)))


def bench_fano_mori_family(seed: int = _SEED + 2891) -> dict[str, float]:
    return _floats(_finite_blob("fano_mori", bench_fano_mori(seed)))


def bench_mmp_algorithm_family(seed: int = _SEED + 2892) -> dict[str, float]:
    return _floats(_finite_blob("mmp_algorithm", bench_mmp_algorithm(seed)))


def bench_toric_flip_family(seed: int = _SEED + 2893) -> dict[str, float]:
    return _floats(_finite_blob("toric_flip", bench_toric_flip(seed)))

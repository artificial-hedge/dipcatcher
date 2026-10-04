"""Wave-524 Ramsey-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.furstenberg import bench_furstenberg
from quant_fund.models.gallai_thm import bench_gallai_thm
from quant_fund.models.hales_jewett import bench_hales_jewett
from quant_fund.models.hindman import bench_hindman
from quant_fund.models.rado_thm import bench_rado_thm
from quant_fund.models.schur_thm import bench_schur_thm

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


def bench_hales_jewett_family(seed: int = _SEED + 3050) -> dict[str, float]:
    return _floats(_finite_blob("hales_jewett", bench_hales_jewett(seed)))


def bench_rado_thm_family(seed: int = _SEED + 3051) -> dict[str, float]:
    return _floats(_finite_blob("rado_thm", bench_rado_thm(seed)))


def bench_gallai_thm_family(seed: int = _SEED + 3052) -> dict[str, float]:
    return _floats(_finite_blob("gallai_thm", bench_gallai_thm(seed)))


def bench_schur_thm_family(seed: int = _SEED + 3053) -> dict[str, float]:
    return _floats(_finite_blob("schur_thm", bench_schur_thm(seed)))


def bench_hindman_family(seed: int = _SEED + 3054) -> dict[str, float]:
    return _floats(_finite_blob("hindman", bench_hindman(seed)))


def bench_furstenberg_family(seed: int = _SEED + 3055) -> dict[str, float]:
    return _floats(_finite_blob("furstenberg", bench_furstenberg(seed)))

"""Wave-397 stochastic-analysis-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bessel3 import bench_bessel3
from quant_fund.models.h_transform import bench_h_transform
from quant_fund.models.occupation_bm import bench_occupation_bm
from quant_fund.models.ost_calcul import bench_ost_calcul
from quant_fund.models.reflect_bm import bench_reflect_bm
from quant_fund.models.tanaka import bench_tanaka

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


def bench_ost_calcul_family(seed: int = _SEED + 2288) -> dict[str, float]:
    return _floats(_finite_blob("ost_calcul", bench_ost_calcul(seed)))


def bench_tanaka_family(seed: int = _SEED + 2289) -> dict[str, float]:
    return _floats(_finite_blob("tanaka", bench_tanaka(seed)))


def bench_bessel3_family(seed: int = _SEED + 2290) -> dict[str, float]:
    return _floats(_finite_blob("bessel3", bench_bessel3(seed)))


def bench_reflect_bm_family(seed: int = _SEED + 2291) -> dict[str, float]:
    return _floats(_finite_blob("reflect_bm", bench_reflect_bm(seed)))


def bench_occupation_bm_family(seed: int = _SEED + 2292) -> dict[str, float]:
    return _floats(_finite_blob("occupation_bm", bench_occupation_bm(seed)))


def bench_h_transform_family(seed: int = _SEED + 2293) -> dict[str, float]:
    return _floats(_finite_blob("h_transform", bench_h_transform(seed)))

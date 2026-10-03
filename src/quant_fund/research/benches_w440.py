"""Wave-440 homotopy-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bott_period import bench_bott_period
from quant_fund.models.hopf_map import bench_hopf_map
from quant_fund.models.postnikov_twr import bench_postnikov_twr
from quant_fund.models.stable_stem import bench_stable_stem
from quant_fund.models.thom_iso import bench_thom_iso
from quant_fund.models.whitehead_twr import bench_whitehead_twr

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


def bench_thom_iso_family(
    seed: int = _SEED + 2546,
) -> dict[str, float]:
    return _floats(_finite_blob("thom_iso", bench_thom_iso(seed)))


def bench_postnikov_twr_family(
    seed: int = _SEED + 2547,
) -> dict[str, float]:
    return _floats(_finite_blob("postnikov_twr", bench_postnikov_twr(seed)))


def bench_whitehead_twr_family(
    seed: int = _SEED + 2548,
) -> dict[str, float]:
    return _floats(_finite_blob("whitehead_twr", bench_whitehead_twr(seed)))


def bench_bott_period_family(
    seed: int = _SEED + 2549,
) -> dict[str, float]:
    return _floats(_finite_blob("bott_period", bench_bott_period(seed)))


def bench_stable_stem_family(
    seed: int = _SEED + 2550,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_stem", bench_stable_stem(seed)))


def bench_hopf_map_family(
    seed: int = _SEED + 2551,
) -> dict[str, float]:
    return _floats(_finite_blob("hopf_map", bench_hopf_map(seed)))

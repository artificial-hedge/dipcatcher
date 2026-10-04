"""Wave-776 point-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.campbell_thm import bench_campbell_thm
from quant_fund.models.cox_process import bench_cox_process
from quant_fund.models.hawkes_point import bench_hawkes_point
from quant_fund.models.marked_point import bench_marked_point
from quant_fund.models.palm_dist import bench_palm_dist
from quant_fund.models.self_excite import bench_self_excite

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


def bench_cox_process_family(
    seed: int = _SEED + 16500,
) -> dict[str, float]:
    return _floats(_finite_blob("cox_process", bench_cox_process(seed)))


def bench_hawkes_point_family(
    seed: int = _SEED + 16501,
) -> dict[str, float]:
    return _floats(_finite_blob("hawkes_point", bench_hawkes_point(seed)))


def bench_self_excite_family(
    seed: int = _SEED + 16502,
) -> dict[str, float]:
    return _floats(_finite_blob("self_excite", bench_self_excite(seed)))


def bench_marked_point_family(
    seed: int = _SEED + 16503,
) -> dict[str, float]:
    return _floats(_finite_blob("marked_point", bench_marked_point(seed)))


def bench_campbell_thm_family(
    seed: int = _SEED + 16504,
) -> dict[str, float]:
    return _floats(_finite_blob("campbell_thm", bench_campbell_thm(seed)))


def bench_palm_dist_family(
    seed: int = _SEED + 16505,
) -> dict[str, float]:
    return _floats(_finite_blob("palm_dist", bench_palm_dist(seed)))

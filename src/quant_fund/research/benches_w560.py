"""Wave-560 geometric-flows bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ancient_solution import bench_ancient_solution
from quant_fund.models.hamilton_ricci import bench_hamilton_ricci
from quant_fund.models.kahler_ricci_flow import bench_kahler_ricci_flow
from quant_fund.models.mean_curvature_flow import (
    bench_mean_curvature_flow,
)
from quant_fund.models.perelman_entropy import bench_perelman_entropy
from quant_fund.models.ricci_soliton import bench_ricci_soliton

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


def bench_hamilton_ricci_family(seed: int = _SEED + 3266) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hamilton_ricci",
            bench_hamilton_ricci(seed),
        )
    )


def bench_perelman_entropy_family(
    seed: int = _SEED + 3267,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "perelman_entropy",
            bench_perelman_entropy(seed),
        )
    )


def bench_ricci_soliton_family(seed: int = _SEED + 3268) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ricci_soliton",
            bench_ricci_soliton(seed),
        )
    )


def bench_kahler_ricci_flow_family(
    seed: int = _SEED + 3269,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kahler_ricci_flow",
            bench_kahler_ricci_flow(seed),
        )
    )


def bench_mean_curvature_flow_family(
    seed: int = _SEED + 3270,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mean_curvature_flow",
            bench_mean_curvature_flow(seed),
        )
    )


def bench_ancient_solution_family(
    seed: int = _SEED + 3271,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ancient_solution",
            bench_ancient_solution(seed),
        )
    )

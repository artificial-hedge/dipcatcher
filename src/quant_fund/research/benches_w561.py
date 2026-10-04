"""Wave-561 minimal-surfaces bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.almgren_pitts import bench_almgren_pitts
from quant_fund.models.brakke_flow import bench_brakke_flow
from quant_fund.models.minimal_surface import bench_minimal_surface
from quant_fund.models.plateau_problem import bench_plateau_problem
from quant_fund.models.simon_regularity import bench_simon_regularity
from quant_fund.models.stable_minimal import bench_stable_minimal

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


def bench_minimal_surface_family(seed: int = _SEED + 3272) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "minimal_surface",
            bench_minimal_surface(seed),
        )
    )


def bench_plateau_problem_family(
    seed: int = _SEED + 3273,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "plateau_problem",
            bench_plateau_problem(seed),
        )
    )


def bench_brakke_flow_family(seed: int = _SEED + 3274) -> dict[str, float]:
    return _floats(_finite_blob("brakke_flow", bench_brakke_flow(seed)))


def bench_almgren_pitts_family(seed: int = _SEED + 3275) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "almgren_pitts",
            bench_almgren_pitts(seed),
        )
    )


def bench_simon_regularity_family(
    seed: int = _SEED + 3276,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "simon_regularity",
            bench_simon_regularity(seed),
        )
    )


def bench_stable_minimal_family(seed: int = _SEED + 3277) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_minimal",
            bench_stable_minimal(seed),
        )
    )

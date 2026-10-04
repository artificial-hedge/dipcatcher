"""Wave-681 homotopy-25 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_factor import bench_homotopy_factor
from quant_fund.models.homotopy_fixed import bench_homotopy_fixed
from quant_fund.models.homotopy_lift import bench_homotopy_lift
from quant_fund.models.homotopy_orbit import bench_homotopy_orbit
from quant_fund.models.stable_operad import bench_stable_operad
from quant_fund.models.stable_sheaf import bench_stable_sheaf

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


def bench_homotopy_lift_family(
    seed: int = _SEED + 7000,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_lift", bench_homotopy_lift(seed)))


def bench_homotopy_orbit_family(
    seed: int = _SEED + 7001,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_orbit", bench_homotopy_orbit(seed)))


def bench_homotopy_fixed_family(
    seed: int = _SEED + 7002,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_fixed", bench_homotopy_fixed(seed)))


def bench_stable_operad_family(
    seed: int = _SEED + 7003,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_operad", bench_stable_operad(seed)))


def bench_homotopy_factor_family(
    seed: int = _SEED + 7004,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_factor", bench_homotopy_factor(seed)))


def bench_stable_sheaf_family(
    seed: int = _SEED + 7005,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_sheaf", bench_stable_sheaf(seed)))

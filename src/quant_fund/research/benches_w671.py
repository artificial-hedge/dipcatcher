"""Wave-671 homotopy-24 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adams_edge import bench_adams_edge
from quant_fund.models.gray_periodic import bench_gray_periodic
from quant_fund.models.homotopy_exponent import (
    bench_homotopy_exponent,
)
from quant_fund.models.periodic_family import (
    bench_periodic_family,
)
from quant_fund.models.stunted_proj import bench_stunted_proj
from quant_fund.models.unstable_adams2 import (
    bench_unstable_adams2,
)

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


def bench_gray_periodic_family(
    seed: int = _SEED + 6000,
) -> dict[str, float]:
    return _floats(_finite_blob("gray_periodic", bench_gray_periodic(seed)))


def bench_stunted_proj_family(
    seed: int = _SEED + 6001,
) -> dict[str, float]:
    return _floats(_finite_blob("stunted_proj", bench_stunted_proj(seed)))


def bench_adams_edge_family(
    seed: int = _SEED + 6002,
) -> dict[str, float]:
    return _floats(_finite_blob("adams_edge", bench_adams_edge(seed)))


def bench_periodic_family_family(
    seed: int = _SEED + 6003,
) -> dict[str, float]:
    return _floats(_finite_blob("periodic_family", bench_periodic_family(seed)))


def bench_unstable_adams2_family(
    seed: int = _SEED + 6004,
) -> dict[str, float]:
    return _floats(_finite_blob("unstable_adams2", bench_unstable_adams2(seed)))


def bench_homotopy_exponent_family(
    seed: int = _SEED + 6005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_exponent",
            bench_homotopy_exponent(seed),
        )
    )

"""Wave-545 Thurston-geometrization bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.eight_geometries import bench_eight_geometries
from quant_fund.models.haken_mfd import bench_haken_mfd
from quant_fund.models.jsj_decomp import bench_jsj_decomp
from quant_fund.models.ricci_flow import bench_ricci_flow
from quant_fund.models.seifert_fibered import bench_seifert_fibered
from quant_fund.models.thurston_geometrization import (
    bench_thurston_geometrization,
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


def bench_thurston_geometrization_family(
    seed: int = _SEED + 3176,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thurston_geometrization",
            bench_thurston_geometrization(seed),
        )
    )


def bench_eight_geometries_family(
    seed: int = _SEED + 3177,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "eight_geometries",
            bench_eight_geometries(seed),
        )
    )


def bench_seifert_fibered_family(
    seed: int = _SEED + 3178,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "seifert_fibered",
            bench_seifert_fibered(seed),
        )
    )


def bench_haken_mfd_family(seed: int = _SEED + 3179) -> dict[str, float]:
    return _floats(_finite_blob("haken_mfd", bench_haken_mfd(seed)))


def bench_jsj_decomp_family(seed: int = _SEED + 3180) -> dict[str, float]:
    return _floats(_finite_blob("jsj_decomp", bench_jsj_decomp(seed)))


def bench_ricci_flow_family(seed: int = _SEED + 3181) -> dict[str, float]:
    return _floats(_finite_blob("ricci_flow", bench_ricci_flow(seed)))

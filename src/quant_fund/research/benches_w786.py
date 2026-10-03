"""Wave-786 rough-path bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.area_mart import bench_area_mart
from quant_fund.models.controlled_path import (
    bench_controlled_path,
)
from quant_fund.models.hairspring_map import (
    bench_hairspring_map,
)
from quant_fund.models.lyons_lift import bench_lyons_lift
from quant_fund.models.rough_path import bench_rough_path
from quant_fund.models.signature_transform2 import (
    bench_signature_transform2,
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


def bench_rough_path_family(
    seed: int = _SEED + 17500,
) -> dict[str, float]:
    return _floats(_finite_blob("rough_path", bench_rough_path(seed)))


def bench_signature_transform2_family(
    seed: int = _SEED + 17501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "signature_transform2",
            bench_signature_transform2(seed),
        )
    )


def bench_controlled_path_family(
    seed: int = _SEED + 17502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "controlled_path",
            bench_controlled_path(seed),
        )
    )


def bench_lyons_lift_family(
    seed: int = _SEED + 17503,
) -> dict[str, float]:
    return _floats(_finite_blob("lyons_lift", bench_lyons_lift(seed)))


def bench_hairspring_map_family(
    seed: int = _SEED + 17504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hairspring_map",
            bench_hairspring_map(seed),
        )
    )


def bench_area_mart_family(
    seed: int = _SEED + 17505,
) -> dict[str, float]:
    return _floats(_finite_blob("area_mart", bench_area_mart(seed)))

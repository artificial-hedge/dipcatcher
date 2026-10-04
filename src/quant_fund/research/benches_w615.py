"""Wave-615 motivic-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.extremal_ray import bench_extremal_ray
from quant_fund.models.mori_bir import bench_mori_bir
from quant_fund.models.motivic_adams import bench_motivic_adams
from quant_fund.models.motivic_classifying import (
    bench_motivic_classifying,
)
from quant_fund.models.motivic_dg import bench_motivic_dg
from quant_fund.models.tate_object import bench_tate_object

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


def bench_motivic_adams_family(
    seed: int = _SEED + 3596,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_adams",
            bench_motivic_adams(seed),
        )
    )


def bench_motivic_classifying_family(
    seed: int = _SEED + 3597,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_classifying",
            bench_motivic_classifying(seed),
        )
    )


def bench_tate_object_family(
    seed: int = _SEED + 3598,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tate_object",
            bench_tate_object(seed),
        )
    )


def bench_motivic_dg_family(
    seed: int = _SEED + 3599,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_dg",
            bench_motivic_dg(seed),
        )
    )


def bench_mori_bir_family(seed: int = _SEED + 3600) -> dict[str, float]:
    return _floats(_finite_blob("mori_bir", bench_mori_bir(seed)))


def bench_extremal_ray_family(
    seed: int = _SEED + 3601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "extremal_ray",
            bench_extremal_ray(seed),
        )
    )

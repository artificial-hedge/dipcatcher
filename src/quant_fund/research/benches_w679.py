"""Wave-679 motivic-17 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.absolute_motive import bench_absolute_motive
from quant_fund.models.etale_motive import bench_etale_motive
from quant_fund.models.motivic_heart import bench_motivic_heart
from quant_fund.models.motivic_realization import (
    bench_motivic_realization,
)
from quant_fund.models.motivic_thh import bench_motivic_thh
from quant_fund.models.relative_motive import bench_relative_motive

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


def bench_motivic_thh_family(
    seed: int = _SEED + 6800,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_thh", bench_motivic_thh(seed)))


def bench_motivic_realization_family(
    seed: int = _SEED + 6801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_realization",
            bench_motivic_realization(seed),
        )
    )


def bench_etale_motive_family(
    seed: int = _SEED + 6802,
) -> dict[str, float]:
    return _floats(_finite_blob("etale_motive", bench_etale_motive(seed)))


def bench_relative_motive_family(
    seed: int = _SEED + 6803,
) -> dict[str, float]:
    return _floats(_finite_blob("relative_motive", bench_relative_motive(seed)))


def bench_absolute_motive_family(
    seed: int = _SEED + 6804,
) -> dict[str, float]:
    return _floats(_finite_blob("absolute_motive", bench_absolute_motive(seed)))


def bench_motivic_heart_family(
    seed: int = _SEED + 6805,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_heart", bench_motivic_heart(seed)))

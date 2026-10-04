"""Wave-689 motivic-20 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_euler import bench_motivic_euler
from quant_fund.models.motivic_ext import bench_motivic_ext
from quant_fund.models.motivic_infinite2 import (
    bench_motivic_infinite2,
)
from quant_fund.models.motivic_jouanolou import (
    bench_motivic_jouanolou,
)
from quant_fund.models.motivic_norm import bench_motivic_norm
from quant_fund.models.motivic_ramified import (
    bench_motivic_ramified,
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


def bench_motivic_jouanolou_family(
    seed: int = _SEED + 7800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_jouanolou",
            bench_motivic_jouanolou(seed),
        )
    )


def bench_motivic_infinite2_family(
    seed: int = _SEED + 7801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_infinite2",
            bench_motivic_infinite2(seed),
        )
    )


def bench_motivic_ext_family(
    seed: int = _SEED + 7802,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_ext", bench_motivic_ext(seed)))


def bench_motivic_norm_family(
    seed: int = _SEED + 7803,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_norm", bench_motivic_norm(seed)))


def bench_motivic_ramified_family(
    seed: int = _SEED + 7804,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_ramified", bench_motivic_ramified(seed)))


def bench_motivic_euler_family(
    seed: int = _SEED + 7805,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_euler", bench_motivic_euler(seed)))

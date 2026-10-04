"""Wave-684 motivic-19 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_base2 import bench_motivic_base2
from quant_fund.models.motivic_frequency import (
    bench_motivic_frequency,
)
from quant_fund.models.motivic_infinite import (
    bench_motivic_infinite,
)
from quant_fund.models.motivic_suslin import bench_motivic_suslin
from quant_fund.models.motivic_weight2 import bench_motivic_weight2
from quant_fund.models.motivic_wit import bench_motivic_wit

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


def bench_motivic_weight2_family(
    seed: int = _SEED + 7300,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_weight2", bench_motivic_weight2(seed)))


def bench_motivic_infinite_family(
    seed: int = _SEED + 7301,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_infinite", bench_motivic_infinite(seed)))


def bench_motivic_suslin_family(
    seed: int = _SEED + 7302,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_suslin", bench_motivic_suslin(seed)))


def bench_motivic_frequency_family(
    seed: int = _SEED + 7303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_frequency",
            bench_motivic_frequency(seed),
        )
    )


def bench_motivic_wit_family(
    seed: int = _SEED + 7304,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_wit", bench_motivic_wit(seed)))


def bench_motivic_base2_family(
    seed: int = _SEED + 7305,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_base2", bench_motivic_base2(seed)))

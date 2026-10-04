"""Wave-701 derived-geometry-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_conn import bench_derived_conn
from quant_fund.models.derived_integral import (
    bench_derived_integral,
)
from quant_fund.models.derived_local import bench_derived_local
from quant_fund.models.derived_noether import (
    bench_derived_noether,
)
from quant_fund.models.derived_normal import bench_derived_normal
from quant_fund.models.derived_reduced import (
    bench_derived_reduced,
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


def bench_derived_conn_family(
    seed: int = _SEED + 9000,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_conn", bench_derived_conn(seed)))


def bench_derived_local_family(
    seed: int = _SEED + 9001,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_local", bench_derived_local(seed)))


def bench_derived_reduced_family(
    seed: int = _SEED + 9002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_reduced",
            bench_derived_reduced(seed),
        )
    )


def bench_derived_integral_family(
    seed: int = _SEED + 9003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_integral",
            bench_derived_integral(seed),
        )
    )


def bench_derived_normal_family(
    seed: int = _SEED + 9004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_normal",
            bench_derived_normal(seed),
        )
    )


def bench_derived_noether_family(
    seed: int = _SEED + 9005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_noether",
            bench_derived_noether(seed),
        )
    )

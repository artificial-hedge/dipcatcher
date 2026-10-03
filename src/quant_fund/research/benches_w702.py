"""Wave-702 motivic-23 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_degree import (
    bench_motivic_degree,
)
from quant_fund.models.motivic_diagonal import (
    bench_motivic_diagonal,
)
from quant_fund.models.motivic_field import bench_motivic_field
from quant_fund.models.motivic_fundamental import (
    bench_motivic_fundamental,
)
from quant_fund.models.motivic_hochschild import (
    bench_motivic_hochschild,
)
from quant_fund.models.motivic_spark import bench_motivic_spark

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


def bench_motivic_spark_family(
    seed: int = _SEED + 9100,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_spark", bench_motivic_spark(seed)))


def bench_motivic_fundamental_family(
    seed: int = _SEED + 9101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_fundamental",
            bench_motivic_fundamental(seed),
        )
    )


def bench_motivic_hochschild_family(
    seed: int = _SEED + 9102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_hochschild",
            bench_motivic_hochschild(seed),
        )
    )


def bench_motivic_field_family(
    seed: int = _SEED + 9103,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_field", bench_motivic_field(seed)))


def bench_motivic_degree_family(
    seed: int = _SEED + 9104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_degree",
            bench_motivic_degree(seed),
        )
    )


def bench_motivic_diagonal_family(
    seed: int = _SEED + 9105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_diagonal",
            bench_motivic_diagonal(seed),
        )
    )

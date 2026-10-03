"""Wave-732 Hall-algebra bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hall_algebra import bench_hall_algebra
from quant_fund.models.joyce_hall import bench_joyce_hall
from quant_fund.models.lusztig_hall import bench_lusztig_hall
from quant_fund.models.ringel_hall import bench_ringel_hall
from quant_fund.models.schiffmann_hall import (
    bench_schiffmann_hall,
)
from quant_fund.models.toen_hall import bench_toen_hall

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


def bench_hall_algebra_family(
    seed: int = _SEED + 12100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hall_algebra",
            bench_hall_algebra(seed),
        )
    )


def bench_ringel_hall_family(
    seed: int = _SEED + 12101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ringel_hall",
            bench_ringel_hall(seed),
        )
    )


def bench_toen_hall_family(
    seed: int = _SEED + 12102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "toen_hall",
            bench_toen_hall(seed),
        )
    )


def bench_lusztig_hall_family(
    seed: int = _SEED + 12103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lusztig_hall",
            bench_lusztig_hall(seed),
        )
    )


def bench_schiffmann_hall_family(
    seed: int = _SEED + 12104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schiffmann_hall",
            bench_schiffmann_hall(seed),
        )
    )


def bench_joyce_hall_family(
    seed: int = _SEED + 12105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "joyce_hall",
            bench_joyce_hall(seed),
        )
    )

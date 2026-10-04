"""Wave-632 witt-vectors-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.big_witt import bench_big_witt
from quant_fund.models.good_reduction import (
    bench_good_reduction,
)
from quant_fund.models.odeur_zarba import (
    bench_odeur_zarba,
)
from quant_fund.models.potential_reduction import (
    bench_potential_reduction,
)
from quant_fund.models.tate_curve import bench_tate_curve
from quant_fund.models.witt_len2 import bench_witt_len2

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


def bench_witt_len2_family(
    seed: int = _SEED + 3698,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "witt_len2",
            bench_witt_len2(seed),
        )
    )


def bench_big_witt_family(
    seed: int = _SEED + 3699,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "big_witt",
            bench_big_witt(seed),
        )
    )


def bench_good_reduction_family(
    seed: int = _SEED + 3700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "good_reduction",
            bench_good_reduction(seed),
        )
    )


def bench_potential_reduction_family(
    seed: int = _SEED + 3701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "potential_reduction",
            bench_potential_reduction(seed),
        )
    )


def bench_tate_curve_family(
    seed: int = _SEED + 3702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tate_curve",
            bench_tate_curve(seed),
        )
    )


def bench_odeur_zarba_family(
    seed: int = _SEED + 3703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "odeur_zarba",
            bench_odeur_zarba(seed),
        )
    )

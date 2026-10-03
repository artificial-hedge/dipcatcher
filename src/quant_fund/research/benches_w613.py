"""Wave-613 arithmetic-geometry-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.neron_smooth import bench_neron_smooth
from quant_fund.models.perfect_witt import bench_perfect_witt
from quant_fund.models.semistable_reduction import (
    bench_semistable_reduction,
)
from quant_fund.models.verschiebung_witt import (
    bench_verschiebung_witt,
)
from quant_fund.models.witt_teich import bench_witt_teich
from quant_fund.models.witt_vector import bench_witt_vector

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


def bench_witt_vector_family(
    seed: int = _SEED + 3584,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "witt_vector",
            bench_witt_vector(seed),
        )
    )


def bench_witt_teich_family(seed: int = _SEED + 3585) -> dict[str, float]:
    return _floats(_finite_blob("witt_teich", bench_witt_teich(seed)))


def bench_verschiebung_witt_family(
    seed: int = _SEED + 3586,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "verschiebung_witt",
            bench_verschiebung_witt(seed),
        )
    )


def bench_perfect_witt_family(
    seed: int = _SEED + 3587,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "perfect_witt",
            bench_perfect_witt(seed),
        )
    )


def bench_neron_smooth_family(
    seed: int = _SEED + 3588,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "neron_smooth",
            bench_neron_smooth(seed),
        )
    )


def bench_semistable_reduction_family(
    seed: int = _SEED + 3589,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "semistable_reduction",
            bench_semistable_reduction(seed),
        )
    )

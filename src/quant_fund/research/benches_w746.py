"""Wave-746 ASEP bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derrida_tasep import bench_derrida_tasep
from quant_fund.models.ferrari_tasep import bench_ferrari_tasep
from quant_fund.models.liggett_exclusion import (
    bench_liggett_exclusion,
)
from quant_fund.models.sasamoto_tasep import bench_sasamoto_tasep
from quant_fund.models.spitzer_exclusion import (
    bench_spitzer_exclusion,
)
from quant_fund.models.tracy_widom_tasep import (
    bench_tracy_widom_tasep,
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


def bench_liggett_exclusion_family(
    seed: int = _SEED + 13500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "liggett_exclusion",
            bench_liggett_exclusion(seed),
        )
    )


def bench_spitzer_exclusion_family(
    seed: int = _SEED + 13501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spitzer_exclusion",
            bench_spitzer_exclusion(seed),
        )
    )


def bench_sasamoto_tasep_family(
    seed: int = _SEED + 13502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sasamoto_tasep",
            bench_sasamoto_tasep(seed),
        )
    )


def bench_tracy_widom_tasep_family(
    seed: int = _SEED + 13503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tracy_widom_tasep",
            bench_tracy_widom_tasep(seed),
        )
    )


def bench_derrida_tasep_family(
    seed: int = _SEED + 13504,
) -> dict[str, float]:
    return _floats(_finite_blob("derrida_tasep", bench_derrida_tasep(seed)))


def bench_ferrari_tasep_family(
    seed: int = _SEED + 13505,
) -> dict[str, float]:
    return _floats(_finite_blob("ferrari_tasep", bench_ferrari_tasep(seed)))

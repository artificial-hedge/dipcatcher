"""Wave-630 infinity-categories-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cartesian_morphism import (
    bench_cartesian_morphism,
)
from quant_fund.models.fib_infty import bench_fib_infty
from quant_fund.models.infty_functor import (
    bench_infty_functor,
)
from quant_fund.models.inner_horn import bench_inner_horn
from quant_fund.models.joyal_horn import bench_joyal_horn
from quant_fund.models.quasi_cat2 import bench_quasi_cat2

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


def bench_quasi_cat2_family(
    seed: int = _SEED + 3686,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quasi_cat2",
            bench_quasi_cat2(seed),
        )
    )


def bench_inner_horn_family(
    seed: int = _SEED + 3687,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "inner_horn",
            bench_inner_horn(seed),
        )
    )


def bench_joyal_horn_family(
    seed: int = _SEED + 3688,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "joyal_horn",
            bench_joyal_horn(seed),
        )
    )


def bench_fib_infty_family(
    seed: int = _SEED + 3689,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fib_infty",
            bench_fib_infty(seed),
        )
    )


def bench_cartesian_morphism_family(
    seed: int = _SEED + 3690,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cartesian_morphism",
            bench_cartesian_morphism(seed),
        )
    )


def bench_infty_functor_family(
    seed: int = _SEED + 3691,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "infty_functor",
            bench_infty_functor(seed),
        )
    )

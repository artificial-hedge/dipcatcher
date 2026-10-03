"""Wave-739 Brownian-map-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bernardi_bijection import (
    bench_bernardi_bijection,
)
from quant_fund.models.bonzom_combe import bench_bonzom_combe
from quant_fund.models.bouttier_guiter import bench_bouttier_guiter
from quant_fund.models.caraceni_curien import bench_caraceni_curien
from quant_fund.models.mullin_bijection import (
    bench_mullin_bijection,
)
from quant_fund.models.schaeffer_bijection import (
    bench_schaeffer_bijection,
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


def bench_caraceni_curien_family(
    seed: int = _SEED + 12800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "caraceni_curien",
            bench_caraceni_curien(seed),
        )
    )


def bench_bonzom_combe_family(
    seed: int = _SEED + 12801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bonzom_combe",
            bench_bonzom_combe(seed),
        )
    )


def bench_mullin_bijection_family(
    seed: int = _SEED + 12802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mullin_bijection",
            bench_mullin_bijection(seed),
        )
    )


def bench_bernardi_bijection_family(
    seed: int = _SEED + 12803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bernardi_bijection",
            bench_bernardi_bijection(seed),
        )
    )


def bench_schaeffer_bijection_family(
    seed: int = _SEED + 12804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schaeffer_bijection",
            bench_schaeffer_bijection(seed),
        )
    )


def bench_bouttier_guiter_family(
    seed: int = _SEED + 12805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bouttier_guiter",
            bench_bouttier_guiter(seed),
        )
    )

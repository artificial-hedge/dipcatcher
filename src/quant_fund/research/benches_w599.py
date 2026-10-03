"""Wave-599 monad-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.algebra_cat import bench_algebra_cat
from quant_fund.models.codensity_monad import (
    bench_codensity_monad,
)
from quant_fund.models.distributive_law import (
    bench_distributive_law,
)
from quant_fund.models.klesli_cat import bench_klesli_cat
from quant_fund.models.monad_theorem import (
    bench_monad_theorem,
)
from quant_fund.models.monadicity import bench_monadicity

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


def bench_monad_theorem_family(
    seed: int = _SEED + 3500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "monad_theorem",
            bench_monad_theorem(seed),
        )
    )


def bench_klesli_cat_family(
    seed: int = _SEED + 3501,
) -> dict[str, float]:
    return _floats(_finite_blob("klesli_cat", bench_klesli_cat(seed)))


def bench_codensity_monad_family(
    seed: int = _SEED + 3502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "codensity_monad",
            bench_codensity_monad(seed),
        )
    )


def bench_monadicity_family(
    seed: int = _SEED + 3503,
) -> dict[str, float]:
    return _floats(_finite_blob("monadicity", bench_monadicity(seed)))


def bench_distributive_law_family(
    seed: int = _SEED + 3504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "distributive_law",
            bench_distributive_law(seed),
        )
    )


def bench_algebra_cat_family(
    seed: int = _SEED + 3505,
) -> dict[str, float]:
    return _floats(_finite_blob("algebra_cat", bench_algebra_cat(seed)))

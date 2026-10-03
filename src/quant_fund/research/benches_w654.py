"""Wave-654 category-11 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.accessible_cat2 import bench_accessible_cat2
from quant_fund.models.compactly_generated import bench_compactly_generated
from quant_fund.models.flat_monad import bench_flat_monad
from quant_fund.models.locally_presentable import bench_locally_presentable
from quant_fund.models.presentable_cat2 import bench_presentable_cat2
from quant_fund.models.regular_cat2 import bench_regular_cat2

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


def bench_compactly_generated_family(
    seed: int = _SEED + 4300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "compactly_generated",
            bench_compactly_generated(seed),
        )
    )


def bench_presentable_cat2_family(
    seed: int = _SEED + 4301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "presentable_cat2",
            bench_presentable_cat2(seed),
        )
    )


def bench_accessible_cat2_family(
    seed: int = _SEED + 4302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "accessible_cat2",
            bench_accessible_cat2(seed),
        )
    )


def bench_flat_monad_family(
    seed: int = _SEED + 4303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "flat_monad",
            bench_flat_monad(seed),
        )
    )


def bench_locally_presentable_family(
    seed: int = _SEED + 4304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "locally_presentable",
            bench_locally_presentable(seed),
        )
    )


def bench_regular_cat2_family(
    seed: int = _SEED + 4305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "regular_cat2",
            bench_regular_cat2(seed),
        )
    )

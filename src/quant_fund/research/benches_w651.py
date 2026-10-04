"""Wave-651 category-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abelian_cat import bench_abelian_cat
from quant_fund.models.filtered_cat import bench_filtered_cat
from quant_fund.models.flat_functor import bench_flat_functor
from quant_fund.models.malcev_cat import bench_malcev_cat
from quant_fund.models.regular_cat import bench_regular_cat
from quant_fund.models.sifted_cat2 import bench_sifted_cat2

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


def bench_flat_functor_family(
    seed: int = _SEED + 4000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "flat_functor",
            bench_flat_functor(seed),
        )
    )


def bench_filtered_cat_family(
    seed: int = _SEED + 4001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "filtered_cat",
            bench_filtered_cat(seed),
        )
    )


def bench_sifted_cat2_family(
    seed: int = _SEED + 4002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sifted_cat2",
            bench_sifted_cat2(seed),
        )
    )


def bench_regular_cat_family(
    seed: int = _SEED + 4003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "regular_cat",
            bench_regular_cat(seed),
        )
    )


def bench_abelian_cat_family(
    seed: int = _SEED + 4004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "abelian_cat",
            bench_abelian_cat(seed),
        )
    )


def bench_malcev_cat_family(
    seed: int = _SEED + 4005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "malcev_cat",
            bench_malcev_cat(seed),
        )
    )

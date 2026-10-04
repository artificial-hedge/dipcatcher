"""Wave-656 category-12 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.admissible_cat import bench_admissible_cat
from quant_fund.models.cartesian_cat2 import bench_cartesian_cat2
from quant_fund.models.cocomplete_cat import bench_cocomplete_cat
from quant_fund.models.definable_cat import bench_definable_cat
from quant_fund.models.essentially_small import bench_essentially_small
from quant_fund.models.finitely_accessible import bench_finitely_accessible

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


def bench_essentially_small_family(
    seed: int = _SEED + 4500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "essentially_small",
            bench_essentially_small(seed),
        )
    )


def bench_finitely_accessible_family(
    seed: int = _SEED + 4501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "finitely_accessible",
            bench_finitely_accessible(seed),
        )
    )


def bench_admissible_cat_family(
    seed: int = _SEED + 4502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "admissible_cat",
            bench_admissible_cat(seed),
        )
    )


def bench_definable_cat_family(
    seed: int = _SEED + 4503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "definable_cat",
            bench_definable_cat(seed),
        )
    )


def bench_cocomplete_cat_family(
    seed: int = _SEED + 4504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cocomplete_cat",
            bench_cocomplete_cat(seed),
        )
    )


def bench_cartesian_cat2_family(
    seed: int = _SEED + 4505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cartesian_cat2",
            bench_cartesian_cat2(seed),
        )
    )

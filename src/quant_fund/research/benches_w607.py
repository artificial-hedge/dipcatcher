"""Wave-607 infinity-categories-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cocartesian import bench_cocartesian
from quant_fund.models.homotopy_cat import (
    bench_homotopy_cat,
)
from quant_fund.models.horn_filler import bench_horn_filler
from quant_fund.models.kan_complex import bench_kan_complex
from quant_fund.models.mapping_space import (
    bench_mapping_space,
)
from quant_fund.models.nerve_cat import bench_nerve_cat

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


def bench_kan_complex_family(
    seed: int = _SEED + 3548,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kan_complex",
            bench_kan_complex(seed),
        )
    )


def bench_horn_filler_family(
    seed: int = _SEED + 3549,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "horn_filler",
            bench_horn_filler(seed),
        )
    )


def bench_nerve_cat_family(seed: int = _SEED + 3550) -> dict[str, float]:
    return _floats(_finite_blob("nerve_cat", bench_nerve_cat(seed)))


def bench_mapping_space_family(
    seed: int = _SEED + 3551,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mapping_space",
            bench_mapping_space(seed),
        )
    )


def bench_homotopy_cat_family(
    seed: int = _SEED + 3552,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_cat",
            bench_homotopy_cat(seed),
        )
    )


def bench_cocartesian_family(
    seed: int = _SEED + 3553,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cocartesian",
            bench_cocartesian(seed),
        )
    )

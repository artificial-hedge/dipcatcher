"""Wave-705 derived-geometry-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_abelian2 import (
    bench_derived_abelian2,
)
from quant_fund.models.derived_cover import bench_derived_cover
from quant_fund.models.derived_geometry7 import (
    bench_derived_geometry7,
)
from quant_fund.models.derived_morph import bench_derived_morph
from quant_fund.models.derived_stack3 import bench_derived_stack3
from quant_fund.models.derived_topos import bench_derived_topos

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


def bench_derived_geometry7_family(
    seed: int = _SEED + 9400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_geometry7",
            bench_derived_geometry7(seed),
        )
    )


def bench_derived_abelian2_family(
    seed: int = _SEED + 9401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_abelian2",
            bench_derived_abelian2(seed),
        )
    )


def bench_derived_stack3_family(
    seed: int = _SEED + 9402,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_stack3", bench_derived_stack3(seed)))


def bench_derived_morph_family(
    seed: int = _SEED + 9403,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_morph", bench_derived_morph(seed)))


def bench_derived_cover_family(
    seed: int = _SEED + 9404,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_cover", bench_derived_cover(seed)))


def bench_derived_topos_family(
    seed: int = _SEED + 9405,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_topos", bench_derived_topos(seed)))

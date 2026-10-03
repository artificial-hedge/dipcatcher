"""Wave-381 algebraic-geometry-6 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ample_test import bench_ample_test
from quant_fund.models.chow_ring import bench_chow_ring
from quant_fund.models.grothendieck_grp import bench_grothendieck_grp
from quant_fund.models.gysin import bench_gysin
from quant_fund.models.proj_morph import bench_proj_morph
from quant_fund.models.toric_variety import bench_toric_variety

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


def bench_grothendieck_grp_family(seed: int = _SEED + 2192) -> dict[str, float]:
    return _floats(_finite_blob("grothendieck_grp", bench_grothendieck_grp(seed)))


def bench_chow_ring_family(seed: int = _SEED + 2193) -> dict[str, float]:
    return _floats(_finite_blob("chow_ring", bench_chow_ring(seed)))


def bench_gysin_family(seed: int = _SEED + 2194) -> dict[str, float]:
    return _floats(_finite_blob("gysin", bench_gysin(seed)))


def bench_toric_variety_family(seed: int = _SEED + 2195) -> dict[str, float]:
    return _floats(_finite_blob("toric_variety", bench_toric_variety(seed)))


def bench_proj_morph_family(seed: int = _SEED + 2196) -> dict[str, float]:
    return _floats(_finite_blob("proj_morph", bench_proj_morph(seed)))


def bench_ample_test_family(seed: int = _SEED + 2197) -> dict[str, float]:
    return _floats(_finite_blob("ample_test", bench_ample_test(seed)))

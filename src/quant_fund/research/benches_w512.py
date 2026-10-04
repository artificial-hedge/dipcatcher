"""Wave-512 super-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.berezin_int import bench_berezin_int
from quant_fund.models.odd_variables import bench_odd_variables
from quant_fund.models.super_lie import bench_super_lie
from quant_fund.models.super_manifold import bench_super_manifold
from quant_fund.models.super_scheme import bench_super_scheme
from quant_fund.models.super_space import bench_super_space

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


def bench_super_space_family(seed: int = _SEED + 2978) -> dict[str, float]:
    return _floats(_finite_blob("super_space", bench_super_space(seed)))


def bench_super_manifold_family(seed: int = _SEED + 2979) -> dict[str, float]:
    return _floats(_finite_blob("super_manifold", bench_super_manifold(seed)))


def bench_super_lie_family(seed: int = _SEED + 2980) -> dict[str, float]:
    return _floats(_finite_blob("super_lie", bench_super_lie(seed)))


def bench_odd_variables_family(seed: int = _SEED + 2981) -> dict[str, float]:
    return _floats(_finite_blob("odd_variables", bench_odd_variables(seed)))


def bench_berezin_int_family(seed: int = _SEED + 2982) -> dict[str, float]:
    return _floats(_finite_blob("berezin_int", bench_berezin_int(seed)))


def bench_super_scheme_family(seed: int = _SEED + 2983) -> dict[str, float]:
    return _floats(_finite_blob("super_scheme", bench_super_scheme(seed)))

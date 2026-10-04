"""Wave-662 higher-algebra-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bar_resolution2 import bench_bar_resolution2
from quant_fund.models.braces_higher import bench_braces_higher
from quant_fund.models.deligne_conj2 import bench_deligne_conj2
from quant_fund.models.factor_homology2 import (
    bench_factor_homology2,
)
from quant_fund.models.hochschild_hom2 import bench_hochschild_hom2
from quant_fund.models.little_cubes import bench_little_cubes

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


def bench_bar_resolution2_family(
    seed: int = _SEED + 5100,
) -> dict[str, float]:
    return _floats(_finite_blob("bar_resolution2", bench_bar_resolution2(seed)))


def bench_hochschild_hom2_family(
    seed: int = _SEED + 5101,
) -> dict[str, float]:
    return _floats(_finite_blob("hochschild_hom2", bench_hochschild_hom2(seed)))


def bench_factor_homology2_family(
    seed: int = _SEED + 5102,
) -> dict[str, float]:
    return _floats(_finite_blob("factor_homology2", bench_factor_homology2(seed)))


def bench_deligne_conj2_family(
    seed: int = _SEED + 5103,
) -> dict[str, float]:
    return _floats(_finite_blob("deligne_conj2", bench_deligne_conj2(seed)))


def bench_braces_higher_family(
    seed: int = _SEED + 5104,
) -> dict[str, float]:
    return _floats(_finite_blob("braces_higher", bench_braces_higher(seed)))


def bench_little_cubes_family(
    seed: int = _SEED + 5105,
) -> dict[str, float]:
    return _floats(_finite_blob("little_cubes", bench_little_cubes(seed)))

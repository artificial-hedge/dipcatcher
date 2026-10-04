"""Wave-503 elliptic-surface bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.elliptic_surface import bench_elliptic_surface
from quant_fund.models.kodaira_fiber import bench_kodaira_fiber
from quant_fund.models.mordell_weil2 import bench_mordell_weil2
from quant_fund.models.neron_model import bench_neron_model
from quant_fund.models.tate_algorithm import bench_tate_algorithm
from quant_fund.models.weierstrass_eq import bench_weierstrass_eq

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


def bench_elliptic_surface_family(seed: int = _SEED + 2924) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_surface", bench_elliptic_surface(seed)))


def bench_weierstrass_eq_family(seed: int = _SEED + 2925) -> dict[str, float]:
    return _floats(_finite_blob("weierstrass_eq", bench_weierstrass_eq(seed)))


def bench_kodaira_fiber_family(seed: int = _SEED + 2926) -> dict[str, float]:
    return _floats(_finite_blob("kodaira_fiber", bench_kodaira_fiber(seed)))


def bench_tate_algorithm_family(seed: int = _SEED + 2927) -> dict[str, float]:
    return _floats(_finite_blob("tate_algorithm", bench_tate_algorithm(seed)))


def bench_mordell_weil2_family(seed: int = _SEED + 2928) -> dict[str, float]:
    return _floats(_finite_blob("mordell_weil2", bench_mordell_weil2(seed)))


def bench_neron_model_family(seed: int = _SEED + 2929) -> dict[str, float]:
    return _floats(_finite_blob("neron_model", bench_neron_model(seed)))

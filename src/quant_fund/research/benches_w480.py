"""Wave-480 infinity-topos-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.classify_obj import bench_classify_obj
from quant_fund.models.etale_geom import bench_etale_geom
from quant_fund.models.exponentiable import bench_exponentiable
from quant_fund.models.gros_topos import bench_gros_topos
from quant_fund.models.local_homeo import bench_local_homeo
from quant_fund.models.pi_infty import bench_pi_infty

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


def bench_etale_geom_family(seed: int = _SEED + 2786) -> dict[str, float]:
    return _floats(_finite_blob("etale_geom", bench_etale_geom(seed)))


def bench_gros_topos_family(seed: int = _SEED + 2787) -> dict[str, float]:
    return _floats(_finite_blob("gros_topos", bench_gros_topos(seed)))


def bench_local_homeo_family(seed: int = _SEED + 2788) -> dict[str, float]:
    return _floats(_finite_blob("local_homeo", bench_local_homeo(seed)))


def bench_classify_obj_family(seed: int = _SEED + 2789) -> dict[str, float]:
    return _floats(_finite_blob("classify_obj", bench_classify_obj(seed)))


def bench_pi_infty_family(seed: int = _SEED + 2790) -> dict[str, float]:
    return _floats(_finite_blob("pi_infty", bench_pi_infty(seed)))


def bench_exponentiable_family(seed: int = _SEED + 2791) -> dict[str, float]:
    return _floats(_finite_blob("exponentiable", bench_exponentiable(seed)))

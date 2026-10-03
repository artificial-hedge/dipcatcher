"""Wave-489 group-theory-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bn_pair import bench_bn_pair
from quant_fund.models.braid_grp import bench_braid_grp
from quant_fund.models.building_toy import bench_building_toy
from quant_fund.models.coxeter_grp import bench_coxeter_grp
from quant_fund.models.hecke_bm import bench_hecke_bm
from quant_fund.models.parabolic_grp import bench_parabolic_grp

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


def bench_building_toy_family(seed: int = _SEED + 2840) -> dict[str, float]:
    return _floats(_finite_blob("building_toy", bench_building_toy(seed)))


def bench_coxeter_grp_family(seed: int = _SEED + 2841) -> dict[str, float]:
    return _floats(_finite_blob("coxeter_grp", bench_coxeter_grp(seed)))


def bench_bn_pair_family(seed: int = _SEED + 2842) -> dict[str, float]:
    return _floats(_finite_blob("bn_pair", bench_bn_pair(seed)))


def bench_braid_grp_family(seed: int = _SEED + 2843) -> dict[str, float]:
    return _floats(_finite_blob("braid_grp", bench_braid_grp(seed)))


def bench_hecke_bm_family(seed: int = _SEED + 2844) -> dict[str, float]:
    return _floats(_finite_blob("hecke_bm", bench_hecke_bm(seed)))


def bench_parabolic_grp_family(seed: int = _SEED + 2845) -> dict[str, float]:
    return _floats(_finite_blob("parabolic_grp", bench_parabolic_grp(seed)))

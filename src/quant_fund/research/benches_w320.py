"""Wave-320 abstract-interpretation canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.affine_karr import bench_affine_karr
from quant_fund.models.andersen_pta import bench_andersen_pta
from quant_fund.models.chaotic_widen import bench_chaotic_widen
from quant_fund.models.interval_analysis import bench_interval_analysis
from quant_fund.models.sign_domain import bench_sign_domain
from quant_fund.models.zone_dbm import bench_zone_dbm

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


def bench_interval_analysis_family(seed: int = _SEED + 1827) -> dict[str, float]:
    return _floats(_finite_blob("interval_analysis", bench_interval_analysis(seed)))


def bench_sign_domain_family(seed: int = _SEED + 1828) -> dict[str, float]:
    return _floats(_finite_blob("sign_domain", bench_sign_domain(seed)))


def bench_zone_dbm_family(seed: int = _SEED + 1829) -> dict[str, float]:
    return _floats(_finite_blob("zone_dbm", bench_zone_dbm(seed)))


def bench_affine_karr_family(seed: int = _SEED + 1830) -> dict[str, float]:
    return _floats(_finite_blob("affine_karr", bench_affine_karr(seed)))


def bench_chaotic_widen_family(seed: int = _SEED + 1831) -> dict[str, float]:
    return _floats(_finite_blob("chaotic_widen", bench_chaotic_widen(seed)))


def bench_andersen_pta_family(seed: int = _SEED + 1832) -> dict[str, float]:
    return _floats(_finite_blob("andersen_pta", bench_andersen_pta(seed)))

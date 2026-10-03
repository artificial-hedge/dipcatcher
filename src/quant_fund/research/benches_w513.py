"""Wave-513 categorification bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.categorify import bench_categorify
from quant_fund.models.hecke_cat import bench_hecke_cat
from quant_fund.models.khovanov_hom import bench_khovanov_hom
from quant_fund.models.rasmussen_inv import bench_rasmussen_inv
from quant_fund.models.soergel_bim import bench_soergel_bim
from quant_fund.models.uq_sl2 import bench_uq_sl2

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


def bench_categorify_family(seed: int = _SEED + 2984) -> dict[str, float]:
    return _floats(_finite_blob("categorify", bench_categorify(seed)))


def bench_khovanov_hom_family(seed: int = _SEED + 2985) -> dict[str, float]:
    return _floats(_finite_blob("khovanov_hom", bench_khovanov_hom(seed)))


def bench_hecke_cat_family(seed: int = _SEED + 2986) -> dict[str, float]:
    return _floats(_finite_blob("hecke_cat", bench_hecke_cat(seed)))


def bench_soergel_bim_family(seed: int = _SEED + 2987) -> dict[str, float]:
    return _floats(_finite_blob("soergel_bim", bench_soergel_bim(seed)))


def bench_rasmussen_inv_family(seed: int = _SEED + 2988) -> dict[str, float]:
    return _floats(_finite_blob("rasmussen_inv", bench_rasmussen_inv(seed)))


def bench_uq_sl2_family(seed: int = _SEED + 2989) -> dict[str, float]:
    return _floats(_finite_blob("uq_sl2", bench_uq_sl2(seed)))

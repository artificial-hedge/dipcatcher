"""Wave-346 number-theory-2/homological-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cohomology_cup import bench_cohomology_cup
from quant_fund.models.elliptic_curve import bench_elliptic_curve
from quant_fund.models.koszul_complex import bench_koszul_complex
from quant_fund.models.mayer_vietoris import bench_mayer_vietoris
from quant_fund.models.p_adic_val import bench_p_adic_val
from quant_fund.models.quadratic_recip import bench_quadratic_recip

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


def bench_quadratic_recip_family(seed: int = _SEED + 1983) -> dict[str, float]:
    return _floats(_finite_blob("quadratic_recip", bench_quadratic_recip(seed)))


def bench_elliptic_curve_family(seed: int = _SEED + 1984) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_curve", bench_elliptic_curve(seed)))


def bench_p_adic_val_family(seed: int = _SEED + 1985) -> dict[str, float]:
    return _floats(_finite_blob("p_adic_val", bench_p_adic_val(seed)))


def bench_cohomology_cup_family(seed: int = _SEED + 1986) -> dict[str, float]:
    return _floats(_finite_blob("cohomology_cup", bench_cohomology_cup(seed)))


def bench_koszul_complex_family(seed: int = _SEED + 1987) -> dict[str, float]:
    return _floats(_finite_blob("koszul_complex", bench_koszul_complex(seed)))


def bench_mayer_vietoris_family(seed: int = _SEED + 1988) -> dict[str, float]:
    return _floats(_finite_blob("mayer_vietoris", bench_mayer_vietoris(seed)))

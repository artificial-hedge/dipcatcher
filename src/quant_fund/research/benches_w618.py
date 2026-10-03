"""Wave-618 p-adic-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ad_period import bench_ad_period
from quant_fund.models.b_drb import bench_b_drb
from quant_fund.models.fontaine_curve import bench_fontaine_curve
from quant_fund.models.perfectoid_c import bench_perfectoid_c
from quant_fund.models.phi_mod import bench_phi_mod
from quant_fund.models.untilt import bench_untilt

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


def bench_fontaine_curve_family(
    seed: int = _SEED + 3614,
) -> dict[str, float]:
    return _floats(_finite_blob("fontaine_curve", bench_fontaine_curve(seed)))


def bench_untilt_family(seed: int = _SEED + 3615) -> dict[str, float]:
    return _floats(_finite_blob("untilt", bench_untilt(seed)))


def bench_perfectoid_c_family(
    seed: int = _SEED + 3616,
) -> dict[str, float]:
    return _floats(_finite_blob("perfectoid_c", bench_perfectoid_c(seed)))


def bench_b_drb_family(seed: int = _SEED + 3617) -> dict[str, float]:
    return _floats(_finite_blob("b_drb", bench_b_drb(seed)))


def bench_phi_mod_family(seed: int = _SEED + 3618) -> dict[str, float]:
    return _floats(_finite_blob("phi_mod", bench_phi_mod(seed)))


def bench_ad_period_family(
    seed: int = _SEED + 3619,
) -> dict[str, float]:
    return _floats(_finite_blob("ad_period", bench_ad_period(seed)))

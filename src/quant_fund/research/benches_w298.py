"""Wave-298 medical-imaging canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.art_sirt import bench_art_sirt
from quant_fund.models.chan_vese import bench_chan_vese
from quant_fund.models.cs_mri import bench_cs_mri
from quant_fund.models.hu_moments import bench_hu_moments
from quant_fund.models.mi_register import bench_mi_register
from quant_fund.models.radon_fbp import bench_radon_fbp

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_radon_fbp_family(seed: int = _SEED + 1694) -> dict[str, float]:
    return _floats(_finite_blob("radon_fbp", bench_radon_fbp(seed)))


def bench_art_sirt_family(seed: int = _SEED + 1695) -> dict[str, float]:
    return _floats(_finite_blob("art_sirt", bench_art_sirt(seed)))


def bench_cs_mri_family(seed: int = _SEED + 1696) -> dict[str, float]:
    return _floats(_finite_blob("cs_mri", bench_cs_mri(seed)))


def bench_hu_moments_family(seed: int = _SEED + 1697) -> dict[str, float]:
    return _floats(_finite_blob("hu_moments", bench_hu_moments(seed)))


def bench_chan_vese_family(seed: int = _SEED + 1698) -> dict[str, float]:
    return _floats(_finite_blob("chan_vese", bench_chan_vese(seed)))


def bench_mi_register_family(seed: int = _SEED + 1699) -> dict[str, float]:
    return _floats(_finite_blob("mi_register", bench_mi_register(seed)))

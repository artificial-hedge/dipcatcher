"""Wave-301 geophysics/seismic canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.avo_shuey import bench_avo_shuey
from quant_fund.models.eikonal_fmm import bench_eikonal_fmm
from quant_fund.models.kirchhoff_mig import bench_kirchhoff_mig
from quant_fund.models.nmo_dix import bench_nmo_dix
from quant_fund.models.taup_transform import bench_taup_transform
from quant_fund.models.vibroseis_sweep import bench_vibroseis_sweep

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


def bench_nmo_dix_family(seed: int = _SEED + 1712) -> dict[str, float]:
    return _floats(_finite_blob("nmo_dix", bench_nmo_dix(seed)))


def bench_taup_transform_family(seed: int = _SEED + 1713) -> dict[str, float]:
    return _floats(_finite_blob("taup_transform", bench_taup_transform(seed)))


def bench_kirchhoff_mig_family(seed: int = _SEED + 1714) -> dict[str, float]:
    return _floats(_finite_blob("kirchhoff_mig", bench_kirchhoff_mig(seed)))


def bench_avo_shuey_family(seed: int = _SEED + 1715) -> dict[str, float]:
    return _floats(_finite_blob("avo_shuey", bench_avo_shuey(seed)))


def bench_vibroseis_sweep_family(seed: int = _SEED + 1716) -> dict[str, float]:
    return _floats(_finite_blob("vibroseis_sweep", bench_vibroseis_sweep(seed)))


def bench_eikonal_fmm_family(seed: int = _SEED + 1717) -> dict[str, float]:
    return _floats(_finite_blob("eikonal_fmm", bench_eikonal_fmm(seed)))

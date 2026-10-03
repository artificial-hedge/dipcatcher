"""Wave-376 homotopy-theory-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cofibration import bench_cofibration
from quant_fund.models.fibration import bench_fibration
from quant_fund.models.serre_ss import bench_serre_ss
from quant_fund.models.spectra import bench_spectra
from quant_fund.models.suspension import bench_suspension
from quant_fund.models.whitehead import bench_whitehead

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


def bench_fibration_family(seed: int = _SEED + 2162) -> dict[str, float]:
    return _floats(_finite_blob("fibration", bench_fibration(seed)))


def bench_cofibration_family(seed: int = _SEED + 2163) -> dict[str, float]:
    return _floats(_finite_blob("cofibration", bench_cofibration(seed)))


def bench_serre_ss_family(seed: int = _SEED + 2164) -> dict[str, float]:
    return _floats(_finite_blob("serre_ss", bench_serre_ss(seed)))


def bench_whitehead_family(seed: int = _SEED + 2165) -> dict[str, float]:
    return _floats(_finite_blob("whitehead", bench_whitehead(seed)))


def bench_suspension_family(seed: int = _SEED + 2166) -> dict[str, float]:
    return _floats(_finite_blob("suspension", bench_suspension(seed)))


def bench_spectra_family(seed: int = _SEED + 2167) -> dict[str, float]:
    return _floats(_finite_blob("spectra", bench_spectra(seed)))

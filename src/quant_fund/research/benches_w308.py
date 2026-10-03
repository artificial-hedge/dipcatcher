"""Wave-308 geophysics-2 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gardner_relation import bench_gardner_relation
from quant_fund.models.gassmann_sub import bench_gassmann_sub
from quant_fund.models.reflectivity_synth import bench_reflectivity_synth
from quant_fund.models.semblance_scan import bench_semblance_scan
from quant_fund.models.spectral_decomp import bench_spectral_decomp
from quant_fund.models.vz_raytrace import bench_vz_raytrace

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


def bench_reflectivity_synth_family(seed: int = _SEED + 1754) -> dict[str, float]:
    return _floats(_finite_blob("reflectivity_synth", bench_reflectivity_synth(seed)))


def bench_gassmann_sub_family(seed: int = _SEED + 1755) -> dict[str, float]:
    return _floats(_finite_blob("gassmann_sub", bench_gassmann_sub(seed)))


def bench_spectral_decomp_family(seed: int = _SEED + 1756) -> dict[str, float]:
    return _floats(_finite_blob("spectral_decomp", bench_spectral_decomp(seed)))


def bench_semblance_scan_family(seed: int = _SEED + 1757) -> dict[str, float]:
    return _floats(_finite_blob("semblance_scan", bench_semblance_scan(seed)))


def bench_gardner_relation_family(seed: int = _SEED + 1758) -> dict[str, float]:
    return _floats(_finite_blob("gardner_relation", bench_gardner_relation(seed)))


def bench_vz_raytrace_family(seed: int = _SEED + 1759) -> dict[str, float]:
    return _floats(_finite_blob("vz_raytrace", bench_vz_raytrace(seed)))

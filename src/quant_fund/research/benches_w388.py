"""Wave-388 homological-algebra-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_functor import bench_derived_functor
from quant_fund.models.ext_compute import bench_ext_compute
from quant_fund.models.koszul_homology import bench_koszul_homology
from quant_fund.models.mapping_degree import bench_mapping_degree
from quant_fund.models.spectral_seq import bench_spectral_seq
from quant_fund.models.tor_compute import bench_tor_compute

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


def bench_derived_functor_family(seed: int = _SEED + 2234) -> dict[str, float]:
    return _floats(_finite_blob("derived_functor", bench_derived_functor(seed)))


def bench_ext_compute_family(seed: int = _SEED + 2235) -> dict[str, float]:
    return _floats(_finite_blob("ext_compute", bench_ext_compute(seed)))


def bench_tor_compute_family(seed: int = _SEED + 2236) -> dict[str, float]:
    return _floats(_finite_blob("tor_compute", bench_tor_compute(seed)))


def bench_spectral_seq_family(seed: int = _SEED + 2237) -> dict[str, float]:
    return _floats(_finite_blob("spectral_seq", bench_spectral_seq(seed)))


def bench_koszul_homology_family(seed: int = _SEED + 2238) -> dict[str, float]:
    return _floats(_finite_blob("koszul_homology", bench_koszul_homology(seed)))


def bench_mapping_degree_family(seed: int = _SEED + 2239) -> dict[str, float]:
    return _floats(_finite_blob("mapping_degree", bench_mapping_degree(seed)))

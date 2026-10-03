"""Wave-510 spectral-AG bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.connective_e_ring import bench_connective_e_ring
from quant_fund.models.elliptic_cohor import bench_elliptic_cohor
from quant_fund.models.spectral_alg import bench_spectral_alg
from quant_fund.models.spectral_scheme2 import bench_spectral_scheme2
from quant_fund.models.spectral_stack import bench_spectral_stack
from quant_fund.models.taf_lurie import bench_taf_lurie

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


def bench_spectral_scheme2_family(seed: int = _SEED + 2966) -> dict[str, float]:
    return _floats(_finite_blob("spectral_scheme2", bench_spectral_scheme2(seed)))


def bench_connective_e_ring_family(seed: int = _SEED + 2967) -> dict[str, float]:
    return _floats(_finite_blob("connective_e_ring", bench_connective_e_ring(seed)))


def bench_spectral_alg_family(seed: int = _SEED + 2968) -> dict[str, float]:
    return _floats(_finite_blob("spectral_alg", bench_spectral_alg(seed)))


def bench_spectral_stack_family(seed: int = _SEED + 2969) -> dict[str, float]:
    return _floats(_finite_blob("spectral_stack", bench_spectral_stack(seed)))


def bench_elliptic_cohor_family(seed: int = _SEED + 2970) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_cohor", bench_elliptic_cohor(seed)))


def bench_taf_lurie_family(seed: int = _SEED + 2971) -> dict[str, float]:
    return _floats(_finite_blob("taf_lurie", bench_taf_lurie(seed)))

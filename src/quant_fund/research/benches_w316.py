"""Wave-316 quantum-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adapt_vqe import bench_adapt_vqe
from quant_fund.models.hhl_lite import bench_hhl_lite
from quant_fund.models.qdrift import bench_qdrift
from quant_fund.models.shadow_tomography import bench_shadow_tomography
from quant_fund.models.trotter_suzuki import bench_trotter_suzuki
from quant_fund.models.vqd_states import bench_vqd_states

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


def bench_trotter_suzuki_family(seed: int = _SEED + 1803) -> dict[str, float]:
    return _floats(_finite_blob("trotter_suzuki", bench_trotter_suzuki(seed)))


def bench_qdrift_family(seed: int = _SEED + 1804) -> dict[str, float]:
    return _floats(_finite_blob("qdrift", bench_qdrift(seed)))


def bench_shadow_tomography_family(seed: int = _SEED + 1805) -> dict[str, float]:
    return _floats(_finite_blob("shadow_tomography", bench_shadow_tomography(seed)))


def bench_vqd_states_family(seed: int = _SEED + 1806) -> dict[str, float]:
    return _floats(_finite_blob("vqd_states", bench_vqd_states(seed)))


def bench_adapt_vqe_family(seed: int = _SEED + 1807) -> dict[str, float]:
    return _floats(_finite_blob("adapt_vqe", bench_adapt_vqe(seed)))


def bench_hhl_lite_family(seed: int = _SEED + 1808) -> dict[str, float]:
    return _floats(_finite_blob("hhl_lite", bench_hhl_lite(seed)))

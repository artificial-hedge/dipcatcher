"""Wave-664 spectral-AG-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.e_ring_moduli import bench_e_ring_moduli
from quant_fund.models.elliptic_spec2 import bench_elliptic_spec2
from quant_fund.models.spectral_artstack import (
    bench_spectral_artstack,
)
from quant_fund.models.spectral_moduli import bench_spectral_moduli
from quant_fund.models.structured_spec import bench_structured_spec
from quant_fund.models.tmf_stack import bench_tmf_stack

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


def bench_spectral_moduli_family(
    seed: int = _SEED + 5300,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_moduli", bench_spectral_moduli(seed)))


def bench_e_ring_moduli_family(
    seed: int = _SEED + 5301,
) -> dict[str, float]:
    return _floats(_finite_blob("e_ring_moduli", bench_e_ring_moduli(seed)))


def bench_tmf_stack_family(
    seed: int = _SEED + 5302,
) -> dict[str, float]:
    return _floats(_finite_blob("tmf_stack", bench_tmf_stack(seed)))


def bench_spectral_artstack_family(
    seed: int = _SEED + 5303,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_artstack", bench_spectral_artstack(seed)))


def bench_structured_spec_family(
    seed: int = _SEED + 5304,
) -> dict[str, float]:
    return _floats(_finite_blob("structured_spec", bench_structured_spec(seed)))


def bench_elliptic_spec2_family(
    seed: int = _SEED + 5305,
) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_spec2", bench_elliptic_spec2(seed)))

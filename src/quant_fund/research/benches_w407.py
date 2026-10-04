"""Wave-407 algebraic-geometry-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adjunction2 import bench_adjunction2
from quant_fund.models.cech_cohom import bench_cech_cohom
from quant_fund.models.flattening import bench_flattening
from quant_fund.models.hilbert_scheme import bench_hilbert_scheme
from quant_fund.models.scheme_fiber import bench_scheme_fiber
from quant_fund.models.serre_duality import bench_serre_duality

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


def bench_cech_cohom_family(seed: int = _SEED + 2348) -> dict[str, float]:
    return _floats(_finite_blob("cech_cohom", bench_cech_cohom(seed)))


def bench_serre_duality_family(
    seed: int = _SEED + 2349,
) -> dict[str, float]:
    return _floats(_finite_blob("serre_duality", bench_serre_duality(seed)))


def bench_adjunction2_family(seed: int = _SEED + 2350) -> dict[str, float]:
    return _floats(_finite_blob("adjunction2", bench_adjunction2(seed)))


def bench_scheme_fiber_family(seed: int = _SEED + 2351) -> dict[str, float]:
    return _floats(_finite_blob("scheme_fiber", bench_scheme_fiber(seed)))


def bench_hilbert_scheme_family(
    seed: int = _SEED + 2352,
) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_scheme", bench_hilbert_scheme(seed)))


def bench_flattening_family(seed: int = _SEED + 2353) -> dict[str, float]:
    return _floats(_finite_blob("flattening", bench_flattening(seed)))

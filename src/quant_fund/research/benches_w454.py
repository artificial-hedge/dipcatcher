"""Wave-454 intersection-cohomology-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.decomp_thm import bench_decomp_thm
from quant_fund.models.fourier_sato import bench_fourier_sato
from quant_fund.models.ic_stalk import bench_ic_stalk
from quant_fund.models.middle_ext import bench_middle_ext
from quant_fund.models.riemann_hilbert import bench_riemann_hilbert
from quant_fund.models.vanishing_cycles import bench_vanishing_cycles

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


def bench_ic_stalk_family(seed: int = _SEED + 2630) -> dict[str, float]:
    return _floats(_finite_blob("ic_stalk", bench_ic_stalk(seed)))


def bench_decomp_thm_family(seed: int = _SEED + 2631) -> dict[str, float]:
    return _floats(_finite_blob("decomp_thm", bench_decomp_thm(seed)))


def bench_riemann_hilbert_family(seed: int = _SEED + 2632) -> dict[str, float]:
    return _floats(_finite_blob("riemann_hilbert", bench_riemann_hilbert(seed)))


def bench_fourier_sato_family(seed: int = _SEED + 2633) -> dict[str, float]:
    return _floats(_finite_blob("fourier_sato", bench_fourier_sato(seed)))


def bench_vanishing_cycles_family(seed: int = _SEED + 2634) -> dict[str, float]:
    return _floats(_finite_blob("vanishing_cycles", bench_vanishing_cycles(seed)))


def bench_middle_ext_family(seed: int = _SEED + 2635) -> dict[str, float]:
    return _floats(_finite_blob("middle_ext", bench_middle_ext(seed)))

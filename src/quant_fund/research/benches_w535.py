"""Wave-535 microlocal-analysis bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.elliptic_est import bench_elliptic_est
from quant_fund.models.fourier_io import bench_fourier_io
from quant_fund.models.propagation_sing import bench_propagation_sing
from quant_fund.models.pseudodiff_op import bench_pseudodiff_op
from quant_fund.models.symbol_calc import bench_symbol_calc
from quant_fund.models.wavefront_set import bench_wavefront_set

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


def bench_wavefront_set_family(seed: int = _SEED + 3116) -> dict[str, float]:
    return _floats(_finite_blob("wavefront_set", bench_wavefront_set(seed)))


def bench_pseudodiff_op_family(seed: int = _SEED + 3117) -> dict[str, float]:
    return _floats(_finite_blob("pseudodiff_op", bench_pseudodiff_op(seed)))


def bench_fourier_io_family(seed: int = _SEED + 3118) -> dict[str, float]:
    return _floats(_finite_blob("fourier_io", bench_fourier_io(seed)))


def bench_symbol_calc_family(seed: int = _SEED + 3119) -> dict[str, float]:
    return _floats(_finite_blob("symbol_calc", bench_symbol_calc(seed)))


def bench_propagation_sing_family(
    seed: int = _SEED + 3120,
) -> dict[str, float]:
    return _floats(_finite_blob("propagation_sing", bench_propagation_sing(seed)))


def bench_elliptic_est_family(seed: int = _SEED + 3121) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_est", bench_elliptic_est(seed)))

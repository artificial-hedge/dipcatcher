"""Wave-575 random-matrix-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.airy_process import bench_airy_process
from quant_fund.models.beta_ensemble import bench_beta_ensemble
from quant_fund.models.circular_law import bench_circular_law
from quant_fund.models.dyson_brownian import (
    bench_dyson_brownian,
)
from quant_fund.models.sine_kernel import bench_sine_kernel
from quant_fund.models.tracy_widom import bench_tracy_widom

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


def bench_circular_law_family(seed: int = _SEED + 3356) -> dict[str, float]:
    return _floats(_finite_blob("circular_law", bench_circular_law(seed)))


def bench_dyson_brownian_family(
    seed: int = _SEED + 3357,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dyson_brownian",
            bench_dyson_brownian(seed),
        )
    )


def bench_sine_kernel_family(seed: int = _SEED + 3358) -> dict[str, float]:
    return _floats(_finite_blob("sine_kernel", bench_sine_kernel(seed)))


def bench_airy_process_family(seed: int = _SEED + 3359) -> dict[str, float]:
    return _floats(_finite_blob("airy_process", bench_airy_process(seed)))


def bench_tracy_widom_family(seed: int = _SEED + 3360) -> dict[str, float]:
    return _floats(_finite_blob("tracy_widom", bench_tracy_widom(seed)))


def bench_beta_ensemble_family(seed: int = _SEED + 3361) -> dict[str, float]:
    return _floats(_finite_blob("beta_ensemble", bench_beta_ensemble(seed)))

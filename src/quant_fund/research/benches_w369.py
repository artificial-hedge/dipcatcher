"""Wave-369 numerical-6 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aitken_delta import bench_aitken_delta
from quant_fund.models.brent_root import bench_brent_root
from quant_fund.models.broyden import bench_broyden
from quant_fund.models.cheb_approx import bench_cheb_approx
from quant_fund.models.collocation_ode import bench_collocation_ode
from quant_fund.models.romberg import bench_romberg

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


def bench_broyden_family(seed: int = _SEED + 2120) -> dict[str, float]:
    return _floats(_finite_blob("broyden", bench_broyden(seed)))


def bench_cheb_approx_family(seed: int = _SEED + 2121) -> dict[str, float]:
    return _floats(_finite_blob("cheb_approx", bench_cheb_approx(seed)))


def bench_brent_root_family(seed: int = _SEED + 2122) -> dict[str, float]:
    return _floats(_finite_blob("brent_root", bench_brent_root(seed)))


def bench_romberg_family(seed: int = _SEED + 2123) -> dict[str, float]:
    return _floats(_finite_blob("romberg", bench_romberg(seed)))


def bench_aitken_delta_family(seed: int = _SEED + 2124) -> dict[str, float]:
    return _floats(_finite_blob("aitken_delta", bench_aitken_delta(seed)))


def bench_collocation_ode_family(seed: int = _SEED + 2125) -> dict[str, float]:
    return _floats(_finite_blob("collocation_ode", bench_collocation_ode(seed)))

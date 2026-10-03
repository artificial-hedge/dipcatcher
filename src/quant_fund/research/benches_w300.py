"""Wave-300 astronomy-2 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.delta_t import bench_delta_t
from quant_fund.models.eclipse_circ import bench_eclipse_circ
from quant_fund.models.equinox_prec import bench_equinox_prec
from quant_fund.models.nutation_lite import bench_nutation_lite
from quant_fund.models.planet_vsop import bench_planet_vsop
from quant_fund.models.rise_set import bench_rise_set

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_equinox_prec_family(seed: int = _SEED + 1706) -> dict[str, float]:
    return _floats(_finite_blob("equinox_prec", bench_equinox_prec(seed)))


def bench_nutation_lite_family(seed: int = _SEED + 1707) -> dict[str, float]:
    return _floats(_finite_blob("nutation_lite", bench_nutation_lite(seed)))


def bench_rise_set_family(seed: int = _SEED + 1708) -> dict[str, float]:
    return _floats(_finite_blob("rise_set", bench_rise_set(seed)))


def bench_eclipse_circ_family(seed: int = _SEED + 1709) -> dict[str, float]:
    return _floats(_finite_blob("eclipse_circ", bench_eclipse_circ(seed)))


def bench_delta_t_family(seed: int = _SEED + 1710) -> dict[str, float]:
    return _floats(_finite_blob("delta_t", bench_delta_t(seed)))


def bench_planet_vsop_family(seed: int = _SEED + 1711) -> dict[str, float]:
    return _floats(_finite_blob("planet_vsop", bench_planet_vsop(seed)))

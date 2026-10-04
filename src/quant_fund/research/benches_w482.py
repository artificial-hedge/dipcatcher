"""Wave-482 chromatic-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.blue_shift import bench_blue_shift
from quant_fund.models.chromatic_fracture import bench_chromatic_fracture
from quant_fund.models.fgsl_group import bench_fgsl_group
from quant_fund.models.morava_stabilizer import bench_morava_stabilizer
from quant_fund.models.red_shift import bench_red_shift
from quant_fund.models.tate_spec import bench_tate_spec

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


def bench_chromatic_fracture_family(seed: int = _SEED + 2798) -> dict[str, float]:
    return _floats(_finite_blob("chromatic_fracture", bench_chromatic_fracture(seed)))


def bench_morava_stabilizer_family(seed: int = _SEED + 2799) -> dict[str, float]:
    return _floats(_finite_blob("morava_stabilizer", bench_morava_stabilizer(seed)))


def bench_fgsl_group_family(seed: int = _SEED + 2800) -> dict[str, float]:
    return _floats(_finite_blob("fgsl_group", bench_fgsl_group(seed)))


def bench_tate_spec_family(seed: int = _SEED + 2801) -> dict[str, float]:
    return _floats(_finite_blob("tate_spec", bench_tate_spec(seed)))


def bench_blue_shift_family(seed: int = _SEED + 2802) -> dict[str, float]:
    return _floats(_finite_blob("blue_shift", bench_blue_shift(seed)))


def bench_red_shift_family(seed: int = _SEED + 2803) -> dict[str, float]:
    return _floats(_finite_blob("red_shift", bench_red_shift(seed)))

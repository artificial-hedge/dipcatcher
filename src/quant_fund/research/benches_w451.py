"""Wave-451 equivariant-homotopy bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.g_spectrum import bench_g_spectrum
from quant_fund.models.mackey_functor import bench_mackey_functor
from quant_fund.models.norm_map import bench_norm_map
from quant_fund.models.ro_grading import bench_ro_grading
from quant_fund.models.tom_dieck import bench_tom_dieck
from quant_fund.models.wirthmuller import bench_wirthmuller

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


def bench_g_spectrum_family(seed: int = _SEED + 2612) -> dict[str, float]:
    return _floats(_finite_blob("g_spectrum", bench_g_spectrum(seed)))


def bench_mackey_functor_family(seed: int = _SEED + 2613) -> dict[str, float]:
    return _floats(_finite_blob("mackey_functor", bench_mackey_functor(seed)))


def bench_norm_map_family(seed: int = _SEED + 2614) -> dict[str, float]:
    return _floats(_finite_blob("norm_map", bench_norm_map(seed)))


def bench_ro_grading_family(seed: int = _SEED + 2615) -> dict[str, float]:
    return _floats(_finite_blob("ro_grading", bench_ro_grading(seed)))


def bench_wirthmuller_family(seed: int = _SEED + 2616) -> dict[str, float]:
    return _floats(_finite_blob("wirthmuller", bench_wirthmuller(seed)))


def bench_tom_dieck_family(seed: int = _SEED + 2617) -> dict[str, float]:
    return _floats(_finite_blob("tom_dieck", bench_tom_dieck(seed)))

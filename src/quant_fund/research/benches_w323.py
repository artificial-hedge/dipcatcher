"""Wave-323 PL-7 effect/session-types canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alg_effects import bench_alg_effects
from quant_fund.models.free_monad import bench_free_monad
from quant_fund.models.gradual_types import bench_gradual_types
from quant_fund.models.row_types import bench_row_types
from quant_fund.models.session_types import bench_session_types
from quant_fund.models.shift_reset import bench_shift_reset

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


def bench_free_monad_family(seed: int = _SEED + 1845) -> dict[str, float]:
    return _floats(_finite_blob("free_monad", bench_free_monad(seed)))


def bench_alg_effects_family(seed: int = _SEED + 1846) -> dict[str, float]:
    return _floats(_finite_blob("alg_effects", bench_alg_effects(seed)))


def bench_shift_reset_family(seed: int = _SEED + 1847) -> dict[str, float]:
    return _floats(_finite_blob("shift_reset", bench_shift_reset(seed)))


def bench_row_types_family(seed: int = _SEED + 1848) -> dict[str, float]:
    return _floats(_finite_blob("row_types", bench_row_types(seed)))


def bench_session_types_family(seed: int = _SEED + 1849) -> dict[str, float]:
    return _floats(_finite_blob("session_types", bench_session_types(seed)))


def bench_gradual_types_family(seed: int = _SEED + 1850) -> dict[str, float]:
    return _floats(_finite_blob("gradual_types", bench_gradual_types(seed)))

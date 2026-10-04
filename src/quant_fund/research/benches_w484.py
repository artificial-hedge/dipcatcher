"""Wave-484 synthetic-math-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.formal_model import bench_formal_model
from quant_fund.models.internal_univ import bench_internal_univ
from quant_fund.models.stein_space import bench_stein_space
from quant_fund.models.synth_stable import bench_synth_stable
from quant_fund.models.univalent_found import bench_univalent_found
from quant_fund.models.virtual_hodge import bench_virtual_hodge

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


def bench_internal_univ_family(seed: int = _SEED + 2810) -> dict[str, float]:
    return _floats(_finite_blob("internal_univ", bench_internal_univ(seed)))


def bench_virtual_hodge_family(seed: int = _SEED + 2811) -> dict[str, float]:
    return _floats(_finite_blob("virtual_hodge", bench_virtual_hodge(seed)))


def bench_stein_space_family(seed: int = _SEED + 2812) -> dict[str, float]:
    return _floats(_finite_blob("stein_space", bench_stein_space(seed)))


def bench_formal_model_family(seed: int = _SEED + 2813) -> dict[str, float]:
    return _floats(_finite_blob("formal_model", bench_formal_model(seed)))


def bench_univalent_found_family(seed: int = _SEED + 2814) -> dict[str, float]:
    return _floats(_finite_blob("univalent_found", bench_univalent_found(seed)))


def bench_synth_stable_family(seed: int = _SEED + 2815) -> dict[str, float]:
    return _floats(_finite_blob("synth_stable", bench_synth_stable(seed)))

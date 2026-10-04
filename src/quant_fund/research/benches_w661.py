"""Wave-661 chromatic-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ambidexterity import bench_ambidexterity
from quant_fund.models.dieudonne_module import bench_dieudonne_module
from quant_fund.models.higher_semiadditivity import (
    bench_higher_semiadditivity,
)
from quant_fund.models.honda_formal import bench_honda_formal
from quant_fund.models.raynaud_height import bench_raynaud_height
from quant_fund.models.tate_height import bench_tate_height

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


def bench_ambidexterity_family(
    seed: int = _SEED + 5000,
) -> dict[str, float]:
    return _floats(_finite_blob("ambidexterity", bench_ambidexterity(seed)))


def bench_higher_semiadditivity_family(
    seed: int = _SEED + 5001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higher_semiadditivity",
            bench_higher_semiadditivity(seed),
        )
    )


def bench_tate_height_family(
    seed: int = _SEED + 5002,
) -> dict[str, float]:
    return _floats(_finite_blob("tate_height", bench_tate_height(seed)))


def bench_dieudonne_module_family(
    seed: int = _SEED + 5003,
) -> dict[str, float]:
    return _floats(_finite_blob("dieudonne_module", bench_dieudonne_module(seed)))


def bench_honda_formal_family(
    seed: int = _SEED + 5004,
) -> dict[str, float]:
    return _floats(_finite_blob("honda_formal", bench_honda_formal(seed)))


def bench_raynaud_height_family(
    seed: int = _SEED + 5005,
) -> dict[str, float]:
    return _floats(_finite_blob("raynaud_height", bench_raynaud_height(seed)))

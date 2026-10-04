"""Wave-621 motivic-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_borel import bench_motivic_borel
from quant_fund.models.motivic_chow import bench_motivic_chow
from quant_fund.models.motivic_class import bench_motivic_class
from quant_fund.models.motivic_height import (
    bench_motivic_height,
)
from quant_fund.models.motivic_homology import (
    bench_motivic_homology,
)
from quant_fund.models.motivic_k import bench_motivic_k

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


def bench_motivic_k_family(
    seed: int = _SEED + 3632,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_k", bench_motivic_k(seed)))


def bench_motivic_borel_family(
    seed: int = _SEED + 3633,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_borel", bench_motivic_borel(seed)))


def bench_motivic_height_family(
    seed: int = _SEED + 3634,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_height", bench_motivic_height(seed)))


def bench_motivic_chow_family(
    seed: int = _SEED + 3635,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_chow", bench_motivic_chow(seed)))


def bench_motivic_homology_family(
    seed: int = _SEED + 3636,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_homology",
            bench_motivic_homology(seed),
        )
    )


def bench_motivic_class_family(
    seed: int = _SEED + 3637,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_class", bench_motivic_class(seed)))

"""Wave-578 arithmetic-geometry-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bogomolov_conj import bench_bogomolov_conj
from quant_fund.models.canonical_height import (
    bench_canonical_height,
)
from quant_fund.models.equidistribution_thm import (
    bench_equidistribution_thm,
)
from quant_fund.models.global_height import bench_global_height
from quant_fund.models.nevanlinna_th import bench_nevanlinna_th
from quant_fund.models.vojta_conj import bench_vojta_conj

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


def bench_global_height_family(seed: int = _SEED + 3374) -> dict[str, float]:
    return _floats(_finite_blob("global_height", bench_global_height(seed)))


def bench_bogomolov_conj_family(
    seed: int = _SEED + 3375,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bogomolov_conj",
            bench_bogomolov_conj(seed),
        )
    )


def bench_equidistribution_thm_family(
    seed: int = _SEED + 3376,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "equidistribution_thm",
            bench_equidistribution_thm(seed),
        )
    )


def bench_canonical_height_family(
    seed: int = _SEED + 3377,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "canonical_height",
            bench_canonical_height(seed),
        )
    )


def bench_nevanlinna_th_family(
    seed: int = _SEED + 3378,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nevanlinna_th",
            bench_nevanlinna_th(seed),
        )
    )


def bench_vojta_conj_family(seed: int = _SEED + 3379) -> dict[str, float]:
    return _floats(_finite_blob("vojta_conj", bench_vojta_conj(seed)))

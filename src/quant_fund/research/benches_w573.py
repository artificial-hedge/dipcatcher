"""Wave-573 Hodge-2/periods bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.absolute_hodge import bench_absolute_hodge
from quant_fund.models.griffiths_transv import (
    bench_griffiths_transv,
)
from quant_fund.models.hodge_class import bench_hodge_class
from quant_fund.models.hodge_conj import bench_hodge_conj
from quant_fund.models.mumford_tate import bench_mumford_tate
from quant_fund.models.period_domain import bench_period_domain

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


def bench_griffiths_transv_family(
    seed: int = _SEED + 3344,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "griffiths_transv",
            bench_griffiths_transv(seed),
        )
    )


def bench_period_domain_family(
    seed: int = _SEED + 3345,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "period_domain",
            bench_period_domain(seed),
        )
    )


def bench_mumford_tate_family(seed: int = _SEED + 3346) -> dict[str, float]:
    return _floats(_finite_blob("mumford_tate", bench_mumford_tate(seed)))


def bench_hodge_class_family(seed: int = _SEED + 3347) -> dict[str, float]:
    return _floats(_finite_blob("hodge_class", bench_hodge_class(seed)))


def bench_absolute_hodge_family(
    seed: int = _SEED + 3348,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "absolute_hodge",
            bench_absolute_hodge(seed),
        )
    )


def bench_hodge_conj_family(seed: int = _SEED + 3349) -> dict[str, float]:
    return _floats(_finite_blob("hodge_conj", bench_hodge_conj(seed)))

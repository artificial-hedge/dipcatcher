"""Wave-737 LQG-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ding_dupias import bench_ding_dupias
from quant_fund.models.gaines_sle import bench_gaines_sle
from quant_fund.models.gwynne_miller import bench_gwynne_miller
from quant_fund.models.miller_wu import bench_miller_wu
from quant_fund.models.rhoade_vargas import bench_rhoade_vargas
from quant_fund.models.sheffield_quantum import (
    bench_sheffield_quantum,
)

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


def bench_sheffield_quantum_family(
    seed: int = _SEED + 12600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sheffield_quantum",
            bench_sheffield_quantum(seed),
        )
    )


def bench_gaines_sle_family(
    seed: int = _SEED + 12601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gaines_sle",
            bench_gaines_sle(seed),
        )
    )


def bench_miller_wu_family(
    seed: int = _SEED + 12602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "miller_wu",
            bench_miller_wu(seed),
        )
    )


def bench_rhoade_vargas_family(
    seed: int = _SEED + 12603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rhoade_vargas",
            bench_rhoade_vargas(seed),
        )
    )


def bench_ding_dupias_family(
    seed: int = _SEED + 12604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ding_dupias",
            bench_ding_dupias(seed),
        )
    )


def bench_gwynne_miller_family(
    seed: int = _SEED + 12605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gwynne_miller",
            bench_gwynne_miller(seed),
        )
    )

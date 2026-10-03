"""Wave-778 loss/vacation-queue bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borel_tanner import bench_borel_tanner
from quant_fund.models.engset import bench_engset
from quant_fund.models.erlang_b import bench_erlang_b
from quant_fund.models.erlang_c import bench_erlang_c
from quant_fund.models.pollaczek_khinchine import (
    bench_pollaczek_khinchine,
)
from quant_fund.models.takacs_vacation import (
    bench_takacs_vacation,
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


def bench_engset_family(
    seed: int = _SEED + 16700,
) -> dict[str, float]:
    return _floats(_finite_blob("engset", bench_engset(seed)))


def bench_erlang_b_family(
    seed: int = _SEED + 16701,
) -> dict[str, float]:
    return _floats(_finite_blob("erlang_b", bench_erlang_b(seed)))


def bench_erlang_c_family(
    seed: int = _SEED + 16702,
) -> dict[str, float]:
    return _floats(_finite_blob("erlang_c", bench_erlang_c(seed)))


def bench_pollaczek_khinchine_family(
    seed: int = _SEED + 16703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pollaczek_khinchine",
            bench_pollaczek_khinchine(seed),
        )
    )


def bench_borel_tanner_family(
    seed: int = _SEED + 16704,
) -> dict[str, float]:
    return _floats(_finite_blob("borel_tanner", bench_borel_tanner(seed)))


def bench_takacs_vacation_family(
    seed: int = _SEED + 16705,
) -> dict[str, float]:
    return _floats(_finite_blob("takacs_vacation", bench_takacs_vacation(seed)))

"""Wave-743 random-matrix-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.baik_rmt import bench_baik_rmt
from quant_fund.models.borodin_olshanski import (
    bench_borodin_olshanski,
)
from quant_fund.models.bourgade_rmt import bench_bourgade_rmt
from quant_fund.models.chafai_rmt import bench_chafai_rmt
from quant_fund.models.cipolloni_erdos import bench_cipolloni_erdos
from quant_fund.models.tao_vu import bench_tao_vu

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


def bench_baik_rmt_family(
    seed: int = _SEED + 13200,
) -> dict[str, float]:
    return _floats(_finite_blob("baik_rmt", bench_baik_rmt(seed)))


def bench_tao_vu_family(
    seed: int = _SEED + 13201,
) -> dict[str, float]:
    return _floats(_finite_blob("tao_vu", bench_tao_vu(seed)))


def bench_borodin_olshanski_family(
    seed: int = _SEED + 13202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "borodin_olshanski",
            bench_borodin_olshanski(seed),
        )
    )


def bench_cipolloni_erdos_family(
    seed: int = _SEED + 13203,
) -> dict[str, float]:
    return _floats(_finite_blob("cipolloni_erdos", bench_cipolloni_erdos(seed)))


def bench_bourgade_rmt_family(
    seed: int = _SEED + 13204,
) -> dict[str, float]:
    return _floats(_finite_blob("bourgade_rmt", bench_bourgade_rmt(seed)))


def bench_chafai_rmt_family(
    seed: int = _SEED + 13205,
) -> dict[str, float]:
    return _floats(_finite_blob("chafai_rmt", bench_chafai_rmt(seed)))

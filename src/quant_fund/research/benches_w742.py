"""Wave-742 random-matrix bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.deift_rmt import bench_deift_rmt
from quant_fund.models.erdos_yau import bench_erdos_yau
from quant_fund.models.forrester_rmt import bench_forrester_rmt
from quant_fund.models.johansson_rmt import bench_johansson_rmt
from quant_fund.models.mehta_rmt import bench_mehta_rmt
from quant_fund.models.soshnikov_rmt import bench_soshnikov_rmt

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


def bench_soshnikov_rmt_family(
    seed: int = _SEED + 13100,
) -> dict[str, float]:
    return _floats(_finite_blob("soshnikov_rmt", bench_soshnikov_rmt(seed)))


def bench_erdos_yau_family(
    seed: int = _SEED + 13101,
) -> dict[str, float]:
    return _floats(_finite_blob("erdos_yau", bench_erdos_yau(seed)))


def bench_forrester_rmt_family(
    seed: int = _SEED + 13102,
) -> dict[str, float]:
    return _floats(_finite_blob("forrester_rmt", bench_forrester_rmt(seed)))


def bench_mehta_rmt_family(
    seed: int = _SEED + 13103,
) -> dict[str, float]:
    return _floats(_finite_blob("mehta_rmt", bench_mehta_rmt(seed)))


def bench_deift_rmt_family(
    seed: int = _SEED + 13104,
) -> dict[str, float]:
    return _floats(_finite_blob("deift_rmt", bench_deift_rmt(seed)))


def bench_johansson_rmt_family(
    seed: int = _SEED + 13105,
) -> dict[str, float]:
    return _floats(_finite_blob("johansson_rmt", bench_johansson_rmt(seed)))

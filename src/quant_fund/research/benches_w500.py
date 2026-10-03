"""Wave-500 arithmetic-Langlands bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.epsilon_factor import bench_epsilon_factor
from quant_fund.models.harris_taylor import bench_harris_taylor
from quant_fund.models.l_packet import bench_l_packet
from quant_fund.models.langlands_functoriality import bench_langlands_functoriality
from quant_fund.models.local_langlands import bench_local_langlands
from quant_fund.models.weil_group import bench_weil_group

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


def bench_local_langlands_family(seed: int = _SEED + 2906) -> dict[str, float]:
    return _floats(_finite_blob("local_langlands", bench_local_langlands(seed)))


def bench_harris_taylor_family(seed: int = _SEED + 2907) -> dict[str, float]:
    return _floats(_finite_blob("harris_taylor", bench_harris_taylor(seed)))


def bench_weil_group_family(seed: int = _SEED + 2908) -> dict[str, float]:
    return _floats(_finite_blob("weil_group", bench_weil_group(seed)))


def bench_langlands_functoriality_family(
    seed: int = _SEED + 2909,
) -> dict[str, float]:
    return _floats(_finite_blob("langlands_functoriality", bench_langlands_functoriality(seed)))


def bench_epsilon_factor_family(seed: int = _SEED + 2910) -> dict[str, float]:
    return _floats(_finite_blob("epsilon_factor", bench_epsilon_factor(seed)))


def bench_l_packet_family(seed: int = _SEED + 2911) -> dict[str, float]:
    return _floats(_finite_blob("l_packet", bench_l_packet(seed)))

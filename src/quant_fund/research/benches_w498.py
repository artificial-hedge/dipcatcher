"""Wave-498 moduli/GW bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gromov_witten import bench_gromov_witten
from quant_fund.models.hilbert_scheme2 import bench_hilbert_scheme2
from quant_fund.models.kuranishi import bench_kuranishi
from quant_fund.models.m_bar_gn import bench_m_bar_gn
from quant_fund.models.quot_scheme import bench_quot_scheme
from quant_fund.models.stable_map import bench_stable_map

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


def bench_kuranishi_family(seed: int = _SEED + 2894) -> dict[str, float]:
    return _floats(_finite_blob("kuranishi", bench_kuranishi(seed)))


def bench_hilbert_scheme2_family(seed: int = _SEED + 2895) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_scheme2", bench_hilbert_scheme2(seed)))


def bench_quot_scheme_family(seed: int = _SEED + 2896) -> dict[str, float]:
    return _floats(_finite_blob("quot_scheme", bench_quot_scheme(seed)))


def bench_m_bar_gn_family(seed: int = _SEED + 2897) -> dict[str, float]:
    return _floats(_finite_blob("m_bar_gn", bench_m_bar_gn(seed)))


def bench_stable_map_family(seed: int = _SEED + 2898) -> dict[str, float]:
    return _floats(_finite_blob("stable_map", bench_stable_map(seed)))


def bench_gromov_witten_family(seed: int = _SEED + 2899) -> dict[str, float]:
    return _floats(_finite_blob("gromov_witten", bench_gromov_witten(seed)))

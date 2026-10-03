"""Wave-486 algebraic-K-theory-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.karoubi_v import bench_karoubi_v
from quant_fund.models.kv_theory import bench_kv_theory
from quant_fund.models.nk_theory import bench_nk_theory
from quant_fund.models.plus_k import bench_plus_k
from quant_fund.models.vorst_stab import bench_vorst_stab
from quant_fund.models.waldhausen_k import bench_waldhausen_k

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


def bench_waldhausen_k_family(seed: int = _SEED + 2822) -> dict[str, float]:
    return _floats(_finite_blob("waldhausen_k", bench_waldhausen_k(seed)))


def bench_plus_k_family(seed: int = _SEED + 2823) -> dict[str, float]:
    return _floats(_finite_blob("plus_k", bench_plus_k(seed)))


def bench_kv_theory_family(seed: int = _SEED + 2824) -> dict[str, float]:
    return _floats(_finite_blob("kv_theory", bench_kv_theory(seed)))


def bench_karoubi_v_family(seed: int = _SEED + 2825) -> dict[str, float]:
    return _floats(_finite_blob("karoubi_v", bench_karoubi_v(seed)))


def bench_vorst_stab_family(seed: int = _SEED + 2826) -> dict[str, float]:
    return _floats(_finite_blob("vorst_stab", bench_vorst_stab(seed)))


def bench_nk_theory_family(seed: int = _SEED + 2827) -> dict[str, float]:
    return _floats(_finite_blob("nk_theory", bench_nk_theory(seed)))

"""Wave-354 algebraic-geometry-4 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dedekind_check import bench_dedekind_check
from quant_fund.models.divisor_group import bench_divisor_group
from quant_fund.models.genus_riemann import bench_genus_riemann
from quant_fund.models.local_ring_zn import bench_local_ring_zn
from quant_fund.models.moduli_naive import bench_moduli_naive
from quant_fund.models.sheaf_gluing import bench_sheaf_gluing

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


def bench_sheaf_gluing_family(seed: int = _SEED + 2031) -> dict[str, float]:
    return _floats(_finite_blob("sheaf_gluing", bench_sheaf_gluing(seed)))


def bench_local_ring_zn_family(seed: int = _SEED + 2032) -> dict[str, float]:
    return _floats(_finite_blob("local_ring_zn", bench_local_ring_zn(seed)))


def bench_dedekind_check_family(seed: int = _SEED + 2033) -> dict[str, float]:
    return _floats(_finite_blob("dedekind_check", bench_dedekind_check(seed)))


def bench_divisor_group_family(seed: int = _SEED + 2034) -> dict[str, float]:
    return _floats(_finite_blob("divisor_group", bench_divisor_group(seed)))


def bench_genus_riemann_family(seed: int = _SEED + 2035) -> dict[str, float]:
    return _floats(_finite_blob("genus_riemann", bench_genus_riemann(seed)))


def bench_moduli_naive_family(seed: int = _SEED + 2036) -> dict[str, float]:
    return _floats(_finite_blob("moduli_naive", bench_moduli_naive(seed)))

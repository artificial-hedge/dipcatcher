"""Wave-345 commutative-algebra/ring-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.minimal_poly import bench_minimal_poly
from quant_fund.models.norm_trace import bench_norm_trace
from quant_fund.models.pid_check import bench_pid_check
from quant_fund.models.quotient_ring import bench_quotient_ring
from quant_fund.models.ring_ideals import bench_ring_ideals
from quant_fund.models.spec_ring import bench_spec_ring

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


def bench_ring_ideals_family(seed: int = _SEED + 1977) -> dict[str, float]:
    return _floats(_finite_blob("ring_ideals", bench_ring_ideals(seed)))


def bench_quotient_ring_family(seed: int = _SEED + 1978) -> dict[str, float]:
    return _floats(_finite_blob("quotient_ring", bench_quotient_ring(seed)))


def bench_pid_check_family(seed: int = _SEED + 1979) -> dict[str, float]:
    return _floats(_finite_blob("pid_check", bench_pid_check(seed)))


def bench_minimal_poly_family(seed: int = _SEED + 1980) -> dict[str, float]:
    return _floats(_finite_blob("minimal_poly", bench_minimal_poly(seed)))


def bench_norm_trace_family(seed: int = _SEED + 1981) -> dict[str, float]:
    return _floats(_finite_blob("norm_trace", bench_norm_trace(seed)))


def bench_spec_ring_family(seed: int = _SEED + 1982) -> dict[str, float]:
    return _floats(_finite_blob("spec_ring", bench_spec_ring(seed)))

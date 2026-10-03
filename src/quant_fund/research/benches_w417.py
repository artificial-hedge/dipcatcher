"""Wave-417 commutative-algebra-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cohen_mac import bench_cohen_mac
from quant_fund.models.depth_ring import bench_depth_ring
from quant_fund.models.free_resolution import bench_free_resolution
from quant_fund.models.groebner_syz import bench_groebner_syz
from quant_fund.models.hilbert_syzygy import bench_hilbert_syzygy
from quant_fund.models.regular_seq import bench_regular_seq

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


def bench_groebner_syz_family(
    seed: int = _SEED + 2408,
) -> dict[str, float]:
    return _floats(_finite_blob("groebner_syz", bench_groebner_syz(seed)))


def bench_free_resolution_family(
    seed: int = _SEED + 2409,
) -> dict[str, float]:
    return _floats(_finite_blob("free_resolution", bench_free_resolution(seed)))


def bench_hilbert_syzygy_family(
    seed: int = _SEED + 2410,
) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_syzygy", bench_hilbert_syzygy(seed)))


def bench_regular_seq_family(
    seed: int = _SEED + 2411,
) -> dict[str, float]:
    return _floats(_finite_blob("regular_seq", bench_regular_seq(seed)))


def bench_depth_ring_family(
    seed: int = _SEED + 2412,
) -> dict[str, float]:
    return _floats(_finite_blob("depth_ring", bench_depth_ring(seed)))


def bench_cohen_mac_family(seed: int = _SEED + 2413) -> dict[str, float]:
    return _floats(_finite_blob("cohen_mac", bench_cohen_mac(seed)))

"""Wave-429 algebraic-K-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bass_heller_swan import (
    bench_bass_heller_swan,
)
from quant_fund.models.k0_group import bench_k0_group
from quant_fund.models.k1_group import bench_k1_group
from quant_fund.models.k_theory_spec import bench_k_theory_spec
from quant_fund.models.milnor_k2 import bench_milnor_k2
from quant_fund.models.quillen_q import bench_quillen_q

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


def bench_k0_group_family(
    seed: int = _SEED + 2480,
) -> dict[str, float]:
    return _floats(_finite_blob("k0_group", bench_k0_group(seed)))


def bench_k1_group_family(
    seed: int = _SEED + 2481,
) -> dict[str, float]:
    return _floats(_finite_blob("k1_group", bench_k1_group(seed)))


def bench_milnor_k2_family(
    seed: int = _SEED + 2482,
) -> dict[str, float]:
    return _floats(_finite_blob("milnor_k2", bench_milnor_k2(seed)))


def bench_quillen_q_family(
    seed: int = _SEED + 2483,
) -> dict[str, float]:
    return _floats(_finite_blob("quillen_q", bench_quillen_q(seed)))


def bench_k_theory_spec_family(
    seed: int = _SEED + 2484,
) -> dict[str, float]:
    return _floats(_finite_blob("k_theory_spec", bench_k_theory_spec(seed)))


def bench_bass_heller_swan_family(
    seed: int = _SEED + 2485,
) -> dict[str, float]:
    return _floats(_finite_blob("bass_heller_swan", bench_bass_heller_swan(seed)))

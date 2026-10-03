"""Wave-399 model-theory-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boolean_prime import bench_boolean_prime
from quant_fund.models.ef_game_toy import bench_ef_game_toy
from quant_fund.models.fraisse_limit import bench_fraisse_limit
from quant_fund.models.qe_dense_order import bench_qe_dense_order
from quant_fund.models.real_closed import bench_real_closed
from quant_fund.models.vaught_test import bench_vaught_test

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


def bench_ef_game_toy_family(seed: int = _SEED + 2300) -> dict[str, float]:
    return _floats(_finite_blob("ef_game_toy", bench_ef_game_toy(seed)))


def bench_vaught_test_family(seed: int = _SEED + 2301) -> dict[str, float]:
    return _floats(_finite_blob("vaught_test", bench_vaught_test(seed)))


def bench_real_closed_family(seed: int = _SEED + 2302) -> dict[str, float]:
    return _floats(_finite_blob("real_closed", bench_real_closed(seed)))


def bench_boolean_prime_family(seed: int = _SEED + 2303) -> dict[str, float]:
    return _floats(_finite_blob("boolean_prime", bench_boolean_prime(seed)))


def bench_fraisse_limit_family(seed: int = _SEED + 2304) -> dict[str, float]:
    return _floats(_finite_blob("fraisse_limit", bench_fraisse_limit(seed)))


def bench_qe_dense_order_family(
    seed: int = _SEED + 2305,
) -> dict[str, float]:
    return _floats(_finite_blob("qe_dense_order", bench_qe_dense_order(seed)))

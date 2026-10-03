"""Wave-421 category-theory-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.endo_coend import bench_endo_coend
from quant_fund.models.frobenius_alg import bench_frobenius_alg
from quant_fund.models.profunctor_toy import bench_profunctor_toy
from quant_fund.models.span_compose import bench_span_compose
from quant_fund.models.star_autonomous import bench_star_autonomous
from quant_fund.models.traced_monoidal import bench_traced_monoidal

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


def bench_traced_monoidal_family(
    seed: int = _SEED + 2432,
) -> dict[str, float]:
    return _floats(_finite_blob("traced_monoidal", bench_traced_monoidal(seed)))


def bench_star_autonomous_family(
    seed: int = _SEED + 2433,
) -> dict[str, float]:
    return _floats(_finite_blob("star_autonomous", bench_star_autonomous(seed)))


def bench_frobenius_alg_family(
    seed: int = _SEED + 2434,
) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_alg", bench_frobenius_alg(seed)))


def bench_span_compose_family(
    seed: int = _SEED + 2435,
) -> dict[str, float]:
    return _floats(_finite_blob("span_compose", bench_span_compose(seed)))


def bench_profunctor_toy_family(
    seed: int = _SEED + 2436,
) -> dict[str, float]:
    return _floats(_finite_blob("profunctor_toy", bench_profunctor_toy(seed)))


def bench_endo_coend_family(
    seed: int = _SEED + 2437,
) -> dict[str, float]:
    return _floats(_finite_blob("endo_coend", bench_endo_coend(seed)))

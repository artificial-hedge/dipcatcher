"""Wave-392 computability canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arithmetization import bench_arithmetization
from quant_fund.models.diagonal_lemma import bench_diagonal_lemma
from quant_fund.models.fixed_point_combinator import bench_fixed_point_combinator
from quant_fund.models.kleene_normal import bench_kleene_normal
from quant_fund.models.mu_recursion import bench_mu_recursion
from quant_fund.models.primitive_recursion import bench_primitive_recursion

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


def bench_mu_recursion_family(seed: int = _SEED + 2258) -> dict[str, float]:
    return _floats(_finite_blob("mu_recursion", bench_mu_recursion(seed)))


def bench_primitive_recursion_family(
    seed: int = _SEED + 2259,
) -> dict[str, float]:
    return _floats(_finite_blob("primitive_recursion", bench_primitive_recursion(seed)))


def bench_diagonal_lemma_family(seed: int = _SEED + 2260) -> dict[str, float]:
    return _floats(_finite_blob("diagonal_lemma", bench_diagonal_lemma(seed)))


def bench_arithmetization_family(seed: int = _SEED + 2261) -> dict[str, float]:
    return _floats(_finite_blob("arithmetization", bench_arithmetization(seed)))


def bench_fixed_point_combinator_family(
    seed: int = _SEED + 2262,
) -> dict[str, float]:
    return _floats(_finite_blob("fixed_point_combinator", bench_fixed_point_combinator(seed)))


def bench_kleene_normal_family(seed: int = _SEED + 2263) -> dict[str, float]:
    return _floats(_finite_blob("kleene_normal", bench_kleene_normal(seed)))

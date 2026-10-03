"""Wave-425 stacks/moduli bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coarse_space import bench_coarse_space
from quant_fund.models.gerbe_toy import bench_gerbe_toy
from quant_fund.models.moduli_stack import bench_moduli_stack
from quant_fund.models.quotient_stack import bench_quotient_stack
from quant_fund.models.stack_morph import bench_stack_morph
from quant_fund.models.stacky_curve import bench_stacky_curve

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


def bench_moduli_stack_family(
    seed: int = _SEED + 2456,
) -> dict[str, float]:
    return _floats(_finite_blob("moduli_stack", bench_moduli_stack(seed)))


def bench_stacky_curve_family(
    seed: int = _SEED + 2457,
) -> dict[str, float]:
    return _floats(_finite_blob("stacky_curve", bench_stacky_curve(seed)))


def bench_coarse_space_family(
    seed: int = _SEED + 2458,
) -> dict[str, float]:
    return _floats(_finite_blob("coarse_space", bench_coarse_space(seed)))


def bench_quotient_stack_family(
    seed: int = _SEED + 2459,
) -> dict[str, float]:
    return _floats(_finite_blob("quotient_stack", bench_quotient_stack(seed)))


def bench_gerbe_toy_family(
    seed: int = _SEED + 2460,
) -> dict[str, float]:
    return _floats(_finite_blob("gerbe_toy", bench_gerbe_toy(seed)))


def bench_stack_morph_family(
    seed: int = _SEED + 2461,
) -> dict[str, float]:
    return _floats(_finite_blob("stack_morph", bench_stack_morph(seed)))

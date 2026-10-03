"""Wave-389 number-fields canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.artin_symbol import bench_artin_symbol
from quant_fund.models.class_group_toy import bench_class_group_toy
from quant_fund.models.decomposition_group import bench_decomposition_group
from quant_fund.models.discriminant_field import bench_discriminant_field
from quant_fund.models.norm_subring import bench_norm_subring
from quant_fund.models.ramification import bench_ramification

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


def bench_norm_subring_family(seed: int = _SEED + 2240) -> dict[str, float]:
    return _floats(_finite_blob("norm_subring", bench_norm_subring(seed)))


def bench_discriminant_field_family(
    seed: int = _SEED + 2241,
) -> dict[str, float]:
    return _floats(_finite_blob("discriminant_field", bench_discriminant_field(seed)))


def bench_decomposition_group_family(
    seed: int = _SEED + 2242,
) -> dict[str, float]:
    return _floats(_finite_blob("decomposition_group", bench_decomposition_group(seed)))


def bench_ramification_family(seed: int = _SEED + 2243) -> dict[str, float]:
    return _floats(_finite_blob("ramification", bench_ramification(seed)))


def bench_artin_symbol_family(seed: int = _SEED + 2244) -> dict[str, float]:
    return _floats(_finite_blob("artin_symbol", bench_artin_symbol(seed)))


def bench_class_group_toy_family(seed: int = _SEED + 2245) -> dict[str, float]:
    return _floats(_finite_blob("class_group_toy", bench_class_group_toy(seed)))

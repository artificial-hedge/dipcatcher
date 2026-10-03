"""Wave-339 algebra canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.field_ext import bench_field_ext
from quant_fund.models.galois_group import bench_galois_group
from quant_fund.models.lie_bracket import bench_lie_bracket
from quant_fund.models.rep_theory import bench_rep_theory
from quant_fund.models.root_system import bench_root_system
from quant_fund.models.splitting_field import bench_splitting_field

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


def bench_field_ext_family(seed: int = _SEED + 1941) -> dict[str, float]:
    return _floats(_finite_blob("field_ext", bench_field_ext(seed)))


def bench_galois_group_family(seed: int = _SEED + 1942) -> dict[str, float]:
    return _floats(_finite_blob("galois_group", bench_galois_group(seed)))


def bench_splitting_field_family(seed: int = _SEED + 1943) -> dict[str, float]:
    return _floats(_finite_blob("splitting_field", bench_splitting_field(seed)))


def bench_lie_bracket_family(seed: int = _SEED + 1944) -> dict[str, float]:
    return _floats(_finite_blob("lie_bracket", bench_lie_bracket(seed)))


def bench_rep_theory_family(seed: int = _SEED + 1945) -> dict[str, float]:
    return _floats(_finite_blob("rep_theory", bench_rep_theory(seed)))


def bench_root_system_family(seed: int = _SEED + 1946) -> dict[str, float]:
    return _floats(_finite_blob("root_system", bench_root_system(seed)))

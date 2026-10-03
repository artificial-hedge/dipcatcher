"""Wave-281 abstract-algebra benches: groups, fields, rings, ideals."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.galois_field import bench_galois_field
from quant_fund.models.group_table import bench_group_table
from quant_fund.models.ideal_member import bench_ideal_member
from quant_fund.models.matrix_grp import bench_matrix_grp
from quant_fund.models.perm_group import bench_perm_group
from quant_fund.models.poly_ring import bench_poly_ring

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_group_table_family(seed: int = _SEED + 1580) -> dict[str, float]:
    return _floats(_finite_blob("group_table", bench_group_table(seed)))


def bench_perm_group_family(seed: int = _SEED + 1581) -> dict[str, float]:
    return _floats(_finite_blob("perm_group", bench_perm_group(seed)))


def bench_galois_field_family(seed: int = _SEED + 1582) -> dict[str, float]:
    return _floats(_finite_blob("galois_field", bench_galois_field(seed)))


def bench_poly_ring_family(seed: int = _SEED + 1583) -> dict[str, float]:
    return _floats(_finite_blob("poly_ring", bench_poly_ring(seed)))


def bench_ideal_member_family(seed: int = _SEED + 1584) -> dict[str, float]:
    return _floats(_finite_blob("ideal_member", bench_ideal_member(seed)))


def bench_matrix_grp_family(seed: int = _SEED + 1585) -> dict[str, float]:
    return _floats(_finite_blob("matrix_grp", bench_matrix_grp(seed)))
